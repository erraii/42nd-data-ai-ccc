#!/usr/bin/env python3
"""Play CYAN in a terminal against a uniformly random MAGENTA opponent."""
from colony_one import ColonyOneEnv


def main():
    env = ColonyOneEnv()
    while not env.terminated:
        print("\n" + env.render())
        if env.current_team == "MAGENTA":
            action = env.sample_legal_action()
            print("Opponent:", env.format_action(action))
        else:
            actions = env.legal_actions()
            for i, action in enumerate(actions):
                print(f"{i}: {env.format_action(action)}")
            choice = input("Move number (q to quit): ").strip()
            if choice.lower() == "q":
                return
            try:
                index = int(choice)
                if not 0 <= index < len(actions):
                    raise ValueError
            except ValueError:
                print("Choose a valid move number from the list.")
                continue
            action = actions[index]
        env.step(action)
    print("\n" + env.render())
    print("Winner:", env.winner)


if __name__ == "__main__":
    main()
