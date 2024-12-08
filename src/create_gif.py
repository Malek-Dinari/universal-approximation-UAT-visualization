import imageio
import os

def create_gif(frames, output_path, fps=15):
    """Combines frames into a GIF."""
    images = [imageio.imread(frame) for frame in frames]
    imageio.mimsave(output_path, images, fps=fps)
    print(f"GIF saved at {output_path}")

if __name__ == "__main__":
    frames = sorted([f"assets/frame_{i + 1}.png" for i in range(1, 201) if os.path.exists(f"assets/frame_{i + 1}.png")])
    create_gif(frames, "assets/universal_approximation.gif")

    # Optional: Cleanup frame files
    for frame in frames:
        os.remove(frame)python src/train_and_visualize.py
        