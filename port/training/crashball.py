import argparse
from collections import deque
from contextlib import ExitStack
import json
import math
from pathlib import Path
import random
import subprocess
import time

import polar as polar_policy


RESPONSE = "@training "
TINY = 5
ACTIONS = (
    (0, 0x0880, 0x0820, 0x8000),
    (0, 0x0820, 0x0880, 0x8000),
    (0, 0x0840, 0x0810, 0x8000),
    (0, 0x0810, 0x0840, 0x8000),
)
WEIGHT_SHAPES = ((34, 68), (68,), (68, 68), (68,), (68, 4), (4,))
POLAR_ACTIONS = (0, 0x10, 0x30, 0x20, 0x60, 0x40, 0xC0, 0x80, 0x90)
POLAR_ACTIONS += tuple(buttons | 0x8000 for buttons in POLAR_ACTIONS)
POLAR_WEIGHTS = ((42, 96), (96,), (96, 96), (96,), (96, len(POLAR_ACTIONS)), (len(POLAR_ACTIONS),))


class Crashball:
    def __init__(self, exe, data, view=False, arena="crashball", hero=0, teacher=False):
        self.arena = arena
        self.hero = hero
        command = [str(exe.resolve()), "--training-view" if view else "--training", "--data", str(data.resolve())]
        if arena == "polar-panic":
            command.append("--training-polar")
        if teacher:
            command.append("--training-teacher")
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )
        try:
            mode = "Polar Panic" if arena == "polar-panic" else "Crashball"
            vblanks = 2 if arena == "polar-panic" else 6
            if self.receive() != {"version": 1, "mode": mode, "vblanks_per_step": vblanks}:
                raise RuntimeError("Unsupported simulator protocol")
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdout.close()

    def receive(self):
        diagnostics = deque(maxlen=8)
        for line in self.process.stdout:
            if line.startswith(RESPONSE):
                result = json.loads(line[len(RESPONSE):])
                if "error" in result:
                    raise RuntimeError(result["error"])
                return result
            diagnostics.append(line.rstrip())
        raise RuntimeError("Crashball stopped before replying\n" + "\n".join(diagnostics))

    def send(self, command):
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def command(self, command):
        self.send(command)
        return self.receive()

    def reset(self, seed, players=1, difficulty=2):
        return self.command(f"reset {seed % 2**32} {players} {difficulty} {self.hero}")

    def send_step(self, *actions):
        buttons = ([POLAR_ACTIONS[action] for action in actions] if self.arena == "polar-panic"
                   else [ACTIONS[i][action] for i, action in enumerate(actions)])
        buttons.extend([0] * (4 - len(buttons)))
        self.send("step " + " ".join(f"{button:x}" for button in buttons))

    def step(self, *actions):
        self.send_step(*actions)
        return self.receive()


def observation(result, player, rng):
    frames = result["frames"]
    if "polar" in frames[-1]:
        frame = frames[-1]
        players = frame["polar"]
        x, z = players[player][:2]
        order = [player] + sorted((i for i in range(4) if i != player), key=lambda i: (
            not players[i][7], (x - players[i][0]) ** 2 + (z - players[i][1]) ** 2, i))
        values = []
        for index in order:
            px, pz, vx, vz, heading, charge, mode, alive = players[index][:8]
            angle = heading * (math.tau / 4096)
            values.extend((px / 2048, pz / 2048, vx / 256, vz / 256, math.sin(angle), math.cos(angle),
                           charge / 256, mode == 0x51, mode in (0x4B, 0x4C, 0x4D), alive))
        minutes = (frame["clock"] >> 16) & 0xFF
        seconds = (frame["clock"] >> 8) & 0xFF
        remaining = 60 * (10 * (minutes >> 4) + (minutes & 15)) + 10 * (seconds >> 4) + (seconds & 15)
        return values + [frame["ice_radius"] / 2048, remaining / 90]
    order = rng.sample(range(5), k=5)
    values = []
    for frame in frames:
        position = frame["position"][player] if frame["score"][player] > 0 else 0xDEAD
        values.append(position * (1 if player % 2 == 0 else -1) / 1201.0)
        balls = list(reversed(frame["balls"])) + [(0, 0)] * (5 - len(frame["balls"]))
        for index in order:
            x, z = balls[index]
            x, z = ((x, z), (-x, -z), (z, -x), (-z, x))[player]
            values.extend((x / 2500.0, z / 2500.0))
    values.append(frames[-1]["kick_ready"][player])
    return values


def load_weights(path, np, shapes=WEIGHT_SHAPES):
    with np.load(path, allow_pickle=False) as data:
        weights = [data[f"arr_{index}"] for index in range(len(shapes))]
    if tuple(weight.shape for weight in weights) != shapes:
        raise ValueError("Unexpected weight shapes for the chosen arena")
    return weights


def convert(args):
    import numpy as np
    import tensorflow as tf

    checkpoint = tf.train.load_checkpoint(str(args.model / "variables" / "variables"))
    weights = [checkpoint.get_tensor(f"layer_with_weights-{layer}/{name}/.ATTRIBUTES/VARIABLE_VALUE")
               for layer in range(3) for name in ("kernel", "bias")]
    if tuple(weight.shape for weight in weights) != WEIGHT_SHAPES:
        raise ValueError("Unexpected RillAi checkpoint weight shapes")
    np.savez(args.output, *weights)


def predict(state, weights, np):
    values = np.asarray(state, dtype=np.float32)
    for index in (0, 2):
        values = np.maximum(values @ weights[index] + weights[index + 1], 0)
    return int(np.argmax(values @ weights[4] + weights[5]))


def score(result):
    return result["frames"][-1]["score"]


def winner(result):
    frame = result["frames"][-1]
    if frame.get("winner", -1) >= 0:
        return frame["winner"]
    survivors = [i for i, value in enumerate(frame["score"]) if value > 0]
    return survivors[0] if len(survivors) == 1 else -1


def simulator(args, view=False, teacher=False):
    return Crashball(args.exe, args.data, view=view, arena=args.arena, hero=args.hero, teacher=teacher)


def player_states(result, players, rng):
    return [observation(result, player, rng) if score(result)[player] > 0 else None
            for player in range(players)]


def polar_edge_penalty(state):
    x, z, vx, vz = state[:4]
    radius = math.hypot(x + 1.25 * vx, z + 1.25 * vz)
    return 0.5 * max(0, radius - 0.65)


def checkpoint_policy(args):
    import numpy as np

    with np.load(args.weights, allow_pickle=False) as checkpoint:
        if "polar_parameters" in checkpoint:
            parameters = checkpoint["polar_parameters"].tolist()
            if args.arena != "polar-panic" or len(parameters) != len(polar_policy.PARAMETER_BOUNDS):
                raise ValueError("Unexpected parameters for the chosen arena")
            return lambda result, rng: polar_policy.action(result, parameters)
    shapes = POLAR_WEIGHTS if args.arena == "polar-panic" else WEIGHT_SHAPES
    weights = load_weights(args.weights, np, shapes)
    return lambda result, rng: predict(observation(result, 0, rng), weights, np)


def measure(args, environments, seeds, choose_action):
    wins = losses = truncated = steps = points = opponents = 0
    count = len(environments)
    for start in range(0, len(seeds), count):
        batch = [(env, random.Random(seed), env.reset(seed, difficulty=args.difficulty))
                 for env, seed in zip(environments, seeds[start:start + count])]
        for _ in range(args.max_steps):
            if not batch:
                break
            for env, rng, result in batch:
                env.send_step(choose_action(result, rng))
            remaining = []
            steps += len(batch)
            for env, rng, _ in batch:
                result = env.receive()
                scores = score(result)
                if scores[0] == 0 or result["done"]:
                    victor = winner(result) if scores[0] else -1
                    wins += victor == 0
                    losses += scores[0] == 0 or victor > 0
                    truncated += scores[0] > 0 and victor == -1
                    points += scores[0]
                    opponents += sum(value > 0 for value in scores[1:])
                else:
                    remaining.append((env, rng, result))
            batch = remaining
        truncated += len(batch)
        points += sum(score(result)[0] for _, _, result in batch)
        opponents += sum(sum(value > 0 for value in score(result)[1:]) for _, _, result in batch)
    return wins, losses, truncated, steps, points, opponents


def evaluate(args):
    action_count = len(POLAR_ACTIONS) if args.arena == "polar-panic" else WEIGHT_SHAPES[-1][0]
    policies = {"idle": lambda result, rng: 0, "random": lambda result, rng: rng.randrange(action_count)}
    if args.weights:
        policies["checkpoint"] = checkpoint_policy(args)
    if args.policy == "builtin":
        policies = {"builtin": policies["idle"]}
    elif args.policy:
        policies = {args.policy: policies[args.policy]}
    started = time.perf_counter()
    seeds = range(args.seed, args.seed + args.episodes)
    with ExitStack() as stack:
        count = min(args.workers, args.episodes)
        environments = [stack.enter_context(simulator(args, teacher=args.policy == "builtin")) for _ in range(count)]
        for policy, choose_action in policies.items():
            wins, losses, truncated, steps, points, _ = measure(args, environments, seeds, choose_action)
            print(json.dumps(dict(arena=args.arena, hero=args.hero, policy=policy, difficulty=args.difficulty, episodes=args.episodes,
                                  wins=wins, losses=losses, truncated=truncated, steps=steps,
                                  mean_score=points / args.episodes)), flush=True)
    print(f"Wall time: {time.perf_counter() - started:.2f}s")


def evolve(args):
    import numpy as np

    rng = random.Random(args.seed)
    seeds = [rng.getrandbits(32) for _ in range(args.episodes)]
    bounds = polar_policy.PARAMETER_BOUNDS
    best = []
    started = time.perf_counter()
    with ExitStack() as stack:
        environments = [stack.enter_context(simulator(args)) for _ in range(min(args.workers, args.episodes))]
        for trial in range(args.trials):
            if trial < 12:
                parameters = [rng.uniform(lo, hi) for lo, hi in bounds]
            else:
                _, parent = rng.choice(best)
                parameters = [min(hi, max(lo, rng.gauss(value, (hi - lo) * 0.18)))
                              for value, (lo, hi) in zip(parent, bounds, strict=True)]
            wins, _, _, steps, _, opponents = measure(
                args, environments, seeds, lambda result, rng: polar_policy.action(result, parameters))
            best = sorted([*best, ((wins, -opponents, steps), parameters)], reverse=True)[:4]
            if best[0][1] is parameters:
                np.savez(args.output, polar_parameters=parameters, hero=args.hero)
                print(f"Trial {trial + 1} won {wins}/{args.episodes} training rounds. Saved {args.output}", flush=True)
    print(f"Tested {args.trials} candidates in {time.perf_counter() - started:.2f}s")


class Replay:
    def __init__(self, capacity, np, inputs):
        self.data = (
            np.empty((capacity, inputs), dtype=np.float32),
            np.empty(capacity, dtype=np.int32),
            np.empty(capacity, dtype=np.float32),
            np.empty((capacity, inputs), dtype=np.float32),
            np.empty(capacity, dtype=np.float32),
        )
        self.capacity = capacity
        self.count = 0
        self.next_index = 0
        self.np = np

    def add(self, *values):
        for array, value in zip(self.data, values, strict=True):
            array[self.next_index] = value
        self.next_index = (self.next_index + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, rng, size):
        indices = self.np.asarray(rng.sample(range(self.count), size))
        if self.count == self.capacity:
            indices = (indices + self.next_index) % self.capacity
        return tuple(array[indices] for array in self.data)


def imitate(args, network, rng, np, tf):
    states = []
    actions = []
    action_by_buttons = {buttons: i for i, buttons in enumerate(POLAR_ACTIONS)}
    with ExitStack() as stack:
        environments = [stack.enter_context(simulator(args, teacher=True)) for _ in range(args.workers)]
        results = [env.reset(rng.getrandbits(32), difficulty=args.difficulty) for env in environments]
        lengths = [0] * len(environments)
        while len(states) < args.demonstrations:
            for env in environments:
                env.send_step(0)
            for i, env in enumerate(environments):
                sample = env.receive()
                buttons = sample["frames"][0]["polar"][0][9] & 0x80F0
                action = action_by_buttons.get(buttons)
                if action is not None and len(states) < args.demonstrations:
                    states.append(observation(results[i], 0, rng))
                    actions.append(action)
                lengths[i] += 1
                if sample["done"] or score(sample)[0] == 0 or lengths[i] >= args.max_steps:
                    sample = env.reset(rng.getrandbits(32), difficulty=args.difficulty)
                    lengths[i] = 0
                results[i] = sample
    print(f"Collected {len(states)} Tiny demonstrations across {len(set(actions))} actions", flush=True)
    network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    network.fit(np.asarray(states, np.float32), np.asarray(actions, np.int32), batch_size=256, epochs=10, verbose=2)
    checkpoint = args.output.with_name(f"{args.output.stem}-imitation.npz")
    np.savez(checkpoint, *network.get_weights(), hero=args.hero)
    print(f"Saved imitation checkpoint to {checkpoint}", flush=True)


def train(args):
    started = time.perf_counter()
    import numpy as np
    import tensorflow as tf

    rng = random.Random(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    polar = args.arena == "polar-panic"
    shapes = POLAR_WEIGHTS if polar else WEIGHT_SHAPES
    inputs, hidden = shapes[0]
    action_count = shapes[-1][0]
    network = tf.keras.Sequential([tf.keras.Input(shape=(inputs,)), tf.keras.layers.Dense(hidden, activation="relu"),
                                   tf.keras.layers.Dense(hidden, activation="relu"), tf.keras.layers.Dense(action_count)])
    if args.weights:
        network.set_weights(load_weights(args.weights, np, shapes))
    if args.demonstrations:
        imitate(args, network, rng, np, tf)
    target = tf.keras.models.clone_model(network)
    target.set_weights(network.get_weights())
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    @tf.function
    def update(states, actions, rewards, next_states, done):
        best = tf.argmax(network(next_states), axis=1, output_type=tf.int32)
        future = tf.gather(target(next_states), best, batch_dims=1)
        expected = tf.stop_gradient(rewards + (0.996 if polar else 0.99) * (1.0 - done) * future)
        with tf.GradientTape() as tape:
            chosen = tf.gather(network(states), actions, batch_dims=1)
            loss = tf.reduce_mean(tf.keras.losses.huber(expected, chosen))
        optimizer.apply_gradients(zip(tape.gradient(loss, network.trainable_weights), network.trainable_weights))
        return tuple(tf.identity(weight) for weight in network.trainable_weights)

    replay = Replay(75_000 if polar else 50_000, np, inputs)
    completed = step = 0
    checkpoint_steps = set(args.checkpoint_step)
    exploration_steps = args.exploration_steps or args.steps
    initial_epsilon = 0.2 if args.weights or args.demonstrations else 1.0
    with ExitStack() as stack:
        count = min(args.workers, args.steps)
        environments = [stack.enter_context(simulator(args)) for _ in range(count)]
        results = [env.reset(args.seed if index == 0 else rng.getrandbits(32), args.players, args.difficulty)
                   for index, env in enumerate(environments)]
        states = [player_states(result, args.players, rng) for result in results]
        episode_steps = [0] * count
        weights = network.get_weights()
        while step < args.steps:
            active = environments[:min(count, args.steps - step)]
            actions = []
            for index, env in enumerate(active):
                epsilon = max(0.05, initial_epsilon * (1 - (step + index + 1) / exploration_steps))
                decisions = []
                for state in states[index]:
                    if state is None:
                        decisions.append(0)
                    elif rng.random() < epsilon:
                        decisions.append(rng.randrange(action_count))
                    else:
                        decisions.append(predict(state, weights, np))
                env.send_step(*decisions)
                actions.append(decisions)
            for index, (env, decisions) in enumerate(zip(active, actions, strict=True)):
                step += 1
                episode_steps[index] += 1
                previous_scores = score(results[index])
                result = env.receive()
                scores = score(result)
                done = result["done"] or not any(value > 0 for value in scores[:args.players])
                following = [None] * args.players if done else player_states(result, args.players, rng)
                for player, (state, action, next_state) in enumerate(zip(states[index], decisions, following, strict=True)):
                    if state is not None:
                        if polar:
                            opponents_eliminated = sum(previous_scores) - sum(scores) - (previous_scores[player] - scores[player])
                            reward = 0.2 * opponents_eliminated + polar_edge_penalty(state)
                            if next_state is None:
                                reward += 2.0 if winner(result) == player else -1.0
                            else:
                                reward -= 0.996 * polar_edge_penalty(next_state) + 0.001
                        else:
                            reward = result["touches"][player] + result["kicks"][player] + scores[player] - previous_scores[player]
                        replay.add(state, action, reward, state if next_state is None else next_state, next_state is None)
                if replay.count >= 256 and step % 4 == 0:
                    weights = [weight.numpy() for weight in update(*replay.sample(rng, 64))]
                if step % 1000 == 0:
                    target.set_weights(weights)
                if step in checkpoint_steps:
                    checkpoint = args.output.with_name(f"{args.output.stem}-{step}.npz")
                    np.savez(checkpoint, *weights, hero=args.hero)
                    print(f"Checkpoint at {step} steps after {time.perf_counter() - started:.2f}s: {checkpoint}", flush=True)
                if done or episode_steps[index] == args.max_steps:
                    completed += done
                    episode_steps[index] = 0
                    if step < args.steps:
                        result = env.reset(rng.getrandbits(32), args.players, args.difficulty)
                        following = player_states(result, args.players, rng)
                results[index] = result
                states[index] = following
    np.savez(args.output, *weights, hero=args.hero)
    print(f"Saved {args.steps} steps and {completed} completed episodes to {args.output}")
    elapsed = time.perf_counter() - started
    print(f"{elapsed:.2f} seconds, {args.steps / elapsed:.0f} training steps/s with {count} simulator workers")


def check(args):
    with simulator(args) as env:
        def rollout(action):
            first = env.reset(args.seed, difficulty=args.difficulty)
            return [first] + [env.step(action) for _ in range(60)]

        left = rollout(1)
        if left != rollout(1):
            raise RuntimeError("The same seed and actions produced different states after reset")
        if left == rollout(2):
            raise RuntimeError("Live controller input did not change the rollout")
        if args.arena == "polar-panic" and any(result["frames"][-1]["polar"][0][8] != TINY for result in left):
            raise RuntimeError("Polar Panic did not load Tiny in seat 1")
        if args.arena == "crashball" and not any(frame["balls"] for result in left for frame in result["frames"]):
            raise RuntimeError("The observation stream never saw an active ball")
    print("Seeded reset, pad input, and arena observations passed")


def watch(args):
    choose_action = checkpoint_policy(args)
    rng = random.Random(args.seed)
    with simulator(args, view=True) as env:
        hero = "Tiny in Player 1's seat" if args.hero == TINY else "Player 1"
        print(f"Watching {args.weights.name} control {hero} against three CPU opponents", flush=True)
        result = env.reset(args.seed, difficulty=args.difficulty)
        try:
            while not result["done"]:
                action = choose_action(result, rng) if score(result)[0] > 0 else 0
                result = env.step(action)
        except (BrokenPipeError, RuntimeError):
            if env.process.wait(timeout=2) == 0:
                return
            raise
        if args.arena == "polar-panic":
            victor = winner(result)
            outcome = f"Player {victor + 1} won" if victor >= 0 else "No winner"
            print(f"{outcome}. Close the game window to exit.", flush=True)
        else:
            print(f"Final scores by seat: {score(result)}. Close the game window to exit.", flush=True)
        if env.process.wait() != 0:
            raise RuntimeError("Crashball exited with an error")


def main():
    parser = argparse.ArgumentParser(description="Train, evaluate, or watch Crashball and Polar Panic policies.")
    commands = parser.add_subparsers(dest="command", required=True)
    conversion = commands.add_parser("convert", help="Convert the published RillAi SavedModel to NumPy weights")
    conversion.add_argument("model", type=Path)
    conversion.add_argument("output", type=Path)
    conversion.set_defaults(run=convert)
    for name, run, help_text in (
        ("check", check, "Verify repeatable resets, controller input, and observations"),
        ("watch", watch, "Watch a saved policy play one round in a game window"),
        ("evaluate", evaluate, "Benchmark idle, random, and optional checkpoint policies on the same seeds"),
        ("train", train, "Train Double DQN from scratch or fine-tune existing weights"),
        ("evolve", evolve, "Tune a Polar Panic policy by playing against the built-in opponents"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--exe", required=True, type=Path)
        command.add_argument("--data", required=True, type=Path)
        command.add_argument("--seed", type=int, default=1)
        command.add_argument("--difficulty", type=int, choices=range(3), default=2)
        if name == "evolve":
            command.set_defaults(arena="polar-panic")
        else:
            command.add_argument("--arena", choices=("crashball", "polar-panic"), default="crashball")
        command.set_defaults(run=run)
        if name == "watch":
            command.add_argument("--weights", type=Path, required=True)
        if name in ("evaluate", "train"):
            command.add_argument("--weights", type=Path)
        if name in ("evaluate", "train", "evolve"):
            command.add_argument("--max-steps", type=int)
            command.add_argument("--workers", type=int, default=1, help="Number of concurrent native simulators")
        if name == "evaluate":
            command.add_argument("--episodes", type=int, default=10)
            command.add_argument("--policy", choices=("idle", "random", "builtin", "checkpoint"), help="Evaluate only this policy")
        if name == "evolve":
            command.add_argument("--episodes", type=int, default=24)
            command.add_argument("--trials", type=int, default=48)
            command.add_argument("--output", required=True, type=Path)
        if name == "train":
            command.add_argument("--steps", type=int, default=20_000)
            command.add_argument("--players", type=int, choices=range(1, 5), default=1, help="Seats controlled by the learning policy in each arena")
            command.add_argument("--checkpoint-step", action="append", type=int, default=[], help="Save an intermediate checkpoint at this step count (repeatable)")
            command.add_argument("--exploration-steps", type=int, help="Steps over which to reduce exploration (default: training budget)")
            command.add_argument("--output", required=True, type=Path)
            command.add_argument("--demonstrations", type=int, default=0, help="Pretrain from Tiny's built-in Polar Panic controller")
    args = parser.parse_args()
    if args.command != "convert":
        args.hero = TINY if args.command == "train" or args.arena == "polar-panic" else 0
        if args.command in ("watch", "evaluate") and args.weights:
            import numpy as np

            with np.load(args.weights, allow_pickle=False) as checkpoint:
                if "hero" in checkpoint:
                    args.hero = int(checkpoint["hero"])
        if args.command in ("evaluate", "train", "evolve") and args.max_steps is None:
            args.max_steps = 3600 if args.arena == "polar-panic" else 3000
    if args.command != "convert" and not 0 <= args.seed < 2**32:
        parser.error("--seed must be an unsigned 32-bit integer")
    for flag in ("episodes", "max_steps", "steps", "trials", "workers"):
        if hasattr(args, flag) and getattr(args, flag) <= 0:
            parser.error(f"--{flag.replace('_', '-')} must be positive")
    if args.command == "evaluate" and args.policy == "checkpoint" and args.weights is None:
        parser.error("--policy checkpoint requires --weights")
    if args.command == "evaluate" and args.policy == "builtin" and args.arena != "polar-panic":
        parser.error("--policy builtin requires Polar Panic")
    if args.command == "train" and (args.demonstrations < 0 or (args.demonstrations and args.arena != "polar-panic")):
        parser.error("--demonstrations requires Polar Panic and a nonnegative count")
    if args.command == "train" and any(not 1 <= step < args.steps for step in args.checkpoint_step):
        parser.error("--checkpoint-step must be between 1 and --steps minus 1")
    if args.command == "train" and args.exploration_steps is not None and args.exploration_steps < 1:
        parser.error("--exploration-steps must be positive")
    args.run(args)


if __name__ == "__main__":
    main()
