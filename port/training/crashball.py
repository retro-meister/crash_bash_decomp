import argparse
from collections import deque
from contextlib import ExitStack
import json
from pathlib import Path
import random
import subprocess
import time


RESPONSE = "@training "
ACTIONS = (
    (0, 0x0880, 0x0820, 0x8000),
    (0, 0x0820, 0x0880, 0x8000),
    (0, 0x0840, 0x0810, 0x8000),
    (0, 0x0810, 0x0840, 0x8000),
)
WEIGHT_SHAPES = ((34, 68), (68,), (68, 68), (68,), (68, 4), (4,))


class Crashball:
    def __init__(self, exe, data):
        self.process = subprocess.Popen(
            [str(exe.resolve()), "--training", "--data", str(data.resolve())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )
        try:
            if self.receive() != {"version": 1, "mode": "Crashball", "vblanks_per_step": 6}:
                raise RuntimeError("Unsupported Crashball simulator protocol")
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
        return self.command(f"reset {seed % 2**32} {players} {difficulty}")

    def send_step(self, *actions):
        buttons = [ACTIONS[i][action] for i, action in enumerate(actions)]
        buttons.extend([0] * (4 - len(buttons)))
        self.send("step " + " ".join(f"{button:x}" for button in buttons))

    def step(self, *actions):
        self.send_step(*actions)
        return self.receive()


def observation(result, player, rng):
    frames = result["frames"]
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


def load_weights(path, np):
    with np.load(path, allow_pickle=False) as data:
        weights = [data[f"arr_{index}"] for index in range(len(WEIGHT_SHAPES))]
    if tuple(weight.shape for weight in weights) != WEIGHT_SHAPES:
        raise ValueError("Unexpected Crashball network weight shapes")
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


def player_states(result, players, rng):
    return [observation(result, player, rng) if score(result)[player] > 0 else None
            for player in range(players)]


def evaluate(args):
    if args.weights:
        import numpy as np

        weights = load_weights(args.weights, np)
    policies = ("idle", "random", "checkpoint") if args.weights else ("idle", "random")
    if args.policy:
        policies = (args.policy,)
    started = time.perf_counter()
    with ExitStack() as stack:
        count = min(args.workers, args.episodes)
        environments = [stack.enter_context(Crashball(args.exe, args.data)) for _ in range(count)]
        end_seed = args.seed + args.episodes
        for policy in policies:
            wins = losses = truncated = steps = points = 0
            for start_seed in range(args.seed, end_seed, count):
                batch = [(env, random.Random(seed), env.reset(seed, difficulty=args.difficulty))
                         for env, seed in zip(environments, range(start_seed, min(end_seed, start_seed + count)))]
                for _ in range(args.max_steps):
                    if not batch:
                        break
                    for env, rng, result in batch:
                        action = 0
                        if policy == "random":
                            action = rng.randrange(4)
                        elif policy == "checkpoint":
                            action = predict(observation(result, 0, rng), weights, np)
                        env.send_step(action)
                    remaining = []
                    steps += len(batch)
                    for env, rng, _ in batch:
                        result = env.receive()
                        scores = score(result)
                        if scores[0] == 0:
                            losses += 1
                        elif result["done"]:
                            won = sum(value > 0 for value in scores) == 1
                            wins += won
                            truncated += not won
                            points += scores[0]
                        else:
                            remaining.append((env, rng, result))
                    batch = remaining
                truncated += len(batch)
                points += sum(score(result)[0] for _, _, result in batch)
            print(json.dumps(dict(policy=policy, difficulty=args.difficulty, episodes=args.episodes,
                                  wins=wins, losses=losses, truncated=truncated, steps=steps,
                                  mean_score=points / args.episodes)), flush=True)
    print(f"Wall time: {time.perf_counter() - started:.2f}s")


class Replay:
    def __init__(self, capacity, np):
        self.data = (
            np.empty((capacity, 34), dtype=np.float32),
            np.empty(capacity, dtype=np.int32),
            np.empty(capacity, dtype=np.float32),
            np.empty((capacity, 34), dtype=np.float32),
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


def train(args):
    started = time.perf_counter()
    import numpy as np
    import tensorflow as tf

    rng = random.Random(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    network = tf.keras.Sequential([
        tf.keras.Input(shape=(34,)),
        tf.keras.layers.Dense(68, activation="relu"),
        tf.keras.layers.Dense(68, activation="relu"),
        tf.keras.layers.Dense(4),
    ])
    if args.weights:
        network.set_weights(load_weights(args.weights, np))
    target = tf.keras.models.clone_model(network)
    target.set_weights(network.get_weights())
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    @tf.function
    def update(states, actions, rewards, next_states, done):
        best = tf.argmax(network(next_states), axis=1, output_type=tf.int32)
        future = tf.gather(target(next_states), best, batch_dims=1)
        expected = tf.stop_gradient(rewards + 0.99 * (1.0 - done) * future)
        with tf.GradientTape() as tape:
            chosen = tf.gather(network(states), actions, batch_dims=1)
            loss = tf.reduce_mean(tf.keras.losses.huber(expected, chosen))
        optimizer.apply_gradients(zip(tape.gradient(loss, network.trainable_weights), network.trainable_weights))
        return tuple(tf.identity(weight) for weight in network.trainable_weights)

    replay = Replay(50_000, np)
    completed = step = 0
    checkpoint_steps = set(args.checkpoint_step)
    exploration_steps = args.exploration_steps or args.steps
    initial_epsilon = 0.2 if args.weights else 1.0
    with ExitStack() as stack:
        count = min(args.workers, args.steps)
        environments = [stack.enter_context(Crashball(args.exe, args.data)) for _ in range(count)]
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
                        decisions.append(rng.randrange(4))
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
                        reward = result["touches"][player] + result["kicks"][player] + scores[player] - previous_scores[player]
                        replay.add(state, action, reward, state if next_state is None else next_state, next_state is None)
                if replay.count >= 256 and step % 4 == 0:
                    weights = [weight.numpy() for weight in update(*replay.sample(rng, 64))]
                if step % 1000 == 0:
                    target.set_weights(weights)
                if step in checkpoint_steps:
                    checkpoint = args.output.with_name(f"{args.output.stem}-{step}.npz")
                    np.savez(checkpoint, *weights)
                    print(f"Checkpoint at {step} steps after {time.perf_counter() - started:.2f}s: {checkpoint}", flush=True)
                if done or episode_steps[index] == args.max_steps:
                    completed += done
                    episode_steps[index] = 0
                    if step < args.steps:
                        result = env.reset(rng.getrandbits(32), args.players, args.difficulty)
                        following = player_states(result, args.players, rng)
                results[index] = result
                states[index] = following
    np.savez(args.output, *weights)
    print(f"Saved {args.steps} steps and {completed} completed episodes to {args.output}")
    elapsed = time.perf_counter() - started
    print(f"{elapsed:.2f} seconds, {args.steps / elapsed:.0f} training steps/s with {count} simulator workers")


def check(args):
    with Crashball(args.exe, args.data) as env:
        def rollout(action):
            first = env.reset(args.seed, difficulty=args.difficulty)
            return [first] + [env.step(action) for _ in range(60)]

        left = rollout(1)
        if left != rollout(1):
            raise RuntimeError("The same seed and actions produced different states after reset")
        if left == rollout(2):
            raise RuntimeError("Live controller input did not change the rollout")
        if not any(frame["balls"] for result in left for frame in result["frames"]):
            raise RuntimeError("The observation stream never saw an active ball")
    print("Seeded reset, pad input, and active ball observations passed")


def main():
    parser = argparse.ArgumentParser(description="Run Crashball rollouts and evaluate or fine-tune RillAi.")
    commands = parser.add_subparsers(dest="command", required=True)
    conversion = commands.add_parser("convert", help="Convert the published RillAi SavedModel to NumPy weights")
    conversion.add_argument("model", type=Path)
    conversion.add_argument("output", type=Path)
    conversion.set_defaults(run=convert)
    for name, run, help_text in (
        ("check", check, "Verify repeatable resets, controller input, and observations"),
        ("evaluate", evaluate, "Benchmark idle, random, and optional checkpoint policies on the same seeds"),
        ("train", train, "Train Double DQN from scratch or fine-tune existing weights"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--exe", required=True, type=Path)
        command.add_argument("--data", required=True, type=Path)
        command.add_argument("--seed", type=int, default=1)
        command.add_argument("--difficulty", type=int, choices=range(3), default=2)
        command.set_defaults(run=run)
        if name in ("evaluate", "train"):
            command.add_argument("--weights", type=Path)
            command.add_argument("--max-steps", type=int, default=3000)
            command.add_argument("--workers", type=int, default=1, help="Number of concurrent native simulators")
        if name == "evaluate":
            command.add_argument("--episodes", type=int, default=10)
            command.add_argument("--policy", choices=("idle", "random", "checkpoint"), help="Evaluate only this policy")
        if name == "train":
            command.add_argument("--steps", type=int, default=20_000)
            command.add_argument("--players", type=int, choices=range(1, 5), default=1, help="Seats controlled by the learning policy in each arena")
            command.add_argument("--checkpoint-step", action="append", type=int, default=[], help="Save an intermediate checkpoint at this step count (repeatable)")
            command.add_argument("--exploration-steps", type=int, help="Steps over which to reduce exploration (default: training budget)")
            command.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command != "convert" and not 0 <= args.seed < 2**32:
        parser.error("--seed must be an unsigned 32-bit integer")
    for flag in ("episodes", "max_steps", "steps", "workers"):
        if hasattr(args, flag) and getattr(args, flag) <= 0:
            parser.error(f"--{flag.replace('_', '-')} must be positive")
    if args.command == "evaluate" and args.policy == "checkpoint" and args.weights is None:
        parser.error("--policy checkpoint requires --weights")
    if args.command == "train" and any(not 1 <= step < args.steps for step in args.checkpoint_step):
        parser.error("--checkpoint-step must be between 1 and --steps minus 1")
    if args.command == "train" and args.exploration_steps is not None and args.exploration_steps < 1:
        parser.error("--exploration-steps must be positive")
    args.run(args)


if __name__ == "__main__":
    main()
