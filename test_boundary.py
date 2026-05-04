from detector.boundary import StrokeCycleBoundaryPolicy
import math

def simulate_stroke(num_cycles=5, frames_per_cycle=30):
    """
    Simulates perfect wrist/shoulder tracking data for a left-to-right swimmer.
    Wrist x oscillates around shoulder x — crosses below shoulder at catch,
    rises above shoulder during recovery.
    """
    frames = []
    shoulder_x = 0.5  # fixed shoulder x (normalized)
    amplitude = 0.15  # how far wrist swings past shoulder

    for i in range(num_cycles * frames_per_cycle):
        # sine wave: starts above shoulder (recovery), dips below (catch)
        t = i / frames_per_cycle
        wrist_x = shoulder_x + amplitude * math.cos(2 * math.pi * t)
        frames.append({
            'frame_idx': i,
            'wrist_x': wrist_x,
            'shoulder_x': shoulder_x
        })

    return frames

def run_test():
    policy = StrokeCycleBoundaryPolicy(debounce_frames=15)
    frames = simulate_stroke(num_cycles=5, frames_per_cycle=30)
    
    boundaries = []
    prev = None

    for frame in frames:
        curr = {'wrist_x': frame['wrist_x'], 'shoulder_x': frame['shoulder_x']}
        if policy.is_boundary(prev, curr, frame['frame_idx']):
            boundaries.append(frame['frame_idx'])
            print(f"Boundary at frame {frame['frame_idx']:03d} | wrist_x: {frame['wrist_x']:.3f} shoulder_x: {frame['shoulder_x']:.3f}")
        prev = curr

    print(f"\nExpected boundaries: 5 (one per cycle)")
    print(f"Detected boundaries: {len(boundaries)}")
    print(f"Boundary frames: {boundaries}")
    print(f"Average frames between boundaries: {(boundaries[-1] - boundaries[0]) / (len(boundaries) - 1):.1f}" if len(boundaries) > 1 else "Not enough boundaries")

if __name__ == "__main__":
    run_test()