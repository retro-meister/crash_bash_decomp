import argparse
from collections import deque
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

    def command(self, command):
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        return self.receive()

    def reset(self, seed, players=1, difficulty=2):
        return self.command(f"reset {seed % 2**32} {players} {difficulty}")

    def step(self, *actions):
        buttons = [ACTIONS[i][action] for i, action in enumerate(actions)]
        buttons.extend([0] * (4 - len(buttons)))
        return self.command("step " + " ".join(f"{button:x}" for button in buttons))


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


def evaluate(args):
    if args.weights:
        import numpy as np

        weights = load_weights(args.weights, np)
    policies = ("idle", "random", "checkpoint") if args.weights else ("idle", "random")
    started = time.perf_counter()
    with Crashball(args.exe, args.data) as env:
        for policy in policies:
            wins = losses = truncated = steps = points = 0
            for seed in range(args.seed, args.seed + args.episodes):
                rng = random.Random(seed)
                result = env.reset(seed, difficulty=args.difficulty)
                for _ in range(args.max_steps):
                    action = 0
                    if policy == "random":
                        action = rng.randrange(4)
                    elif policy == "checkpoint":
                        action = predict(observation(result, 0, rng), weights, np)
                    result = env.step(action)
                    steps += 1
                    if score(result)[0] == 0:
                        losses += 1
                        break
                    if result["done"]:
                        wins += sum(value > 0 for value in score(result)) == 1
                        truncated += sum(value > 0 for value in score(result)) != 1
                        break
                else:
                    truncated += 1
                points += score(result)[0]
            print(json.dumps(dict(policy=policy, difficulty=args.difficulty, episodes=args.episodes,
                                  wins=wins, losses=losses, truncated=truncated, steps=steps,
                                  mean_score=points / args.episodes)), flush=True)
    print(f"Wall time: {time.perf_counter() - started:.2f}s")


def train(args):
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

    replay = deque(maxlen=50_000)
    completed = episode_steps = 0
    with Crashball(args.exe, args.data) as env:
        result = env.reset(args.seed, difficulty=args.difficulty)
        weights = network.get_weights()
        state = observation(result, 0, rng)
        for step in range(1, args.steps + 1):
            epsilon = max(0.05, (0.2 if args.weights else 1.0) * (1 - step / args.steps))
            action = rng.randrange(4) if rng.random() < epsilon else predict(state, weights, np)
            previous_score = score(result)[0]
            result = env.step(action)
            episode_steps += 1
            current_score = score(result)[0]
            done = current_score == 0 or result["done"]
            next_state = state if done else observation(result, 0, rng)
            reward = result["touches"][0] + result["kicks"][0] + current_score - previous_score
            replay.append((state, action, reward, next_state, done))
            state = next_state
            if len(replay) >= 256 and step % 4 == 0:
                states, actions, rewards, next_states, dones = zip(*rng.sample(replay, 64))
                update(np.asarray(states, np.float32), np.asarray(actions, np.int32), np.asarray(rewards, np.float32),
                       np.asarray(next_states, np.float32), np.asarray(dones, np.float32))
                weights = network.get_weights()
            if step % 1000 == 0:
                target.set_weights(weights)
            if done or episode_steps == args.max_steps:
                completed += done
                episode_steps = 0
                if step < args.steps:
                    result = env.reset(rng.getrandbits(32), difficulty=args.difficulty)
                    state = observation(result, 0, rng)
    np.savez(args.output, *network.get_weights())
    print(f"Saved {args.steps} steps and {completed} completed episodes to {args.output}")


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
        if name == "evaluate":
            command.add_argument("--episodes", type=int, default=10)
        if name == "train":
            command.add_argument("--steps", type=int, default=20_000)
            command.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command != "convert" and not 0 <= args.seed < 2**32:
        parser.error("--seed must be an unsigned 32-bit integer")
    for flag in ("episodes", "max_steps", "steps"):
        if hasattr(args, flag) and getattr(args, flag) <= 0:
            parser.error(f"--{flag.replace('_', '-')} must be positive")
    args.run(args)


if __name__ == "__main__":
    main()
