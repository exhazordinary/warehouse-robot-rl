def print_grid(grid):
    print("\n".join(" ".join(row) for row in grid))


def print_episode_info(episode, total_reward):
    print(f"\n🏁 Episode {episode + 1} finished with total reward: {total_reward}")
