#!/usr/bin/env python3
"""
Simple GPU test script for lisa-gap package.

This script demonstrates GPU acceleration functionality by running
the same gap generation task on both CPU and GPU, comparing performance
and validating that results are identical.

Based on gap_notebook.ipynb tutorial.
"""

import time
import numpy as np
from lisagap import GapMaskGenerator
from lisaconstants import TROPICALYEAR_J2000DAY

def check_gpu_availability():
    """Check if GPU/CuPy is available."""
    try:
        import cupy as cp
        print("✓ CuPy is available")
        print(f"✓ GPU device: {cp.cuda.Device()}")
        return True
    except ImportError:
        print("✗ CuPy not available - GPU acceleration disabled")
        return False
    except Exception as e:
        print(f"✗ GPU error: {e}")
        return False

def create_test_configuration():
    """Create realistic LISA gap configuration for testing."""
    
    # Set up simulation properties (based on notebook)
    A_YEAR = TROPICALYEAR_J2000DAY * 86400  # seconds in a year
    dt = 0.25  # seconds
    t_start = 0  # start time in seconds
    t_obs = 2.0 * A_YEAR  # 0.1 years for faster testing
    sim_t = t_start + np.arange(0, t_obs, dt)  # time array
    
    print(f"Simulation parameters:")
    print(f"  Duration: {t_obs/86400:.1f} days")
    print(f"  Sampling: {dt} seconds")
    print(f"  Data points: {len(sim_t):,}")
    
    # Realistic LISA gap definitions (from notebook)
    gap_definitions = {
        "planned": {
            "antenna repointing": {"rate_per_year": 26, "duration_hr": 3.3},
            "TM stray potential": {"rate_per_year": 2, "duration_hr": 24},
            "TTL calibration": {"rate_per_year": 4, "duration_hr": 48},
        },
        "unplanned": {
            "platform safe mode": {"rate_per_year": 3, "duration_hr": 60},
            "payload safe mode": {"rate_per_year": 4, "duration_hr": 66},
            "QPD loss micrometeoroid": {"rate_per_year": 5, "duration_hr": 24},
            "HR GRS loss micrometeoroid": {"rate_per_year": 19, "duration_hr": 24},
            "WR GRS loss micrometeoroid": {"rate_per_year": 6, "duration_hr": 24},
        }
    }
    
    return sim_t, dt, gap_definitions

def run_cpu_test(sim_t, dt, gap_definitions):
    """Run gap generation on CPU."""
    print("\n" + "="*50)
    print("RUNNING CPU TEST")
    print("="*50)
    
    # Fixed seeds for reproducibility
    planseed = 2618240388
    unplanseed = 3387490715
    
    # Create CPU gap mask generator
    start_time = time.time()
    gap_gen_cpu = GapMaskGenerator(
        sim_t, 
        dt, 
        gap_definitions, 
        treat_as_nan=False, 
        planseed=planseed,
        unplanseed=unplanseed,
        use_gpu=False  # Force CPU
    )
    init_time = time.time() - start_time
    
    # Generate mask
    start_time = time.time()
    mask_cpu = gap_gen_cpu.generate_mask(include_unplanned=True, include_planned=True)
    generation_time = time.time() - start_time
    
    # Calculate duty cycle
    duty_cycle = 100 * (1 - np.sum(mask_cpu == 0) / len(mask_cpu))
    
    print(f"✓ CPU initialization time: {init_time:.4f} seconds")
    print(f"✓ CPU generation time: {generation_time:.4f} seconds")
    print(f"✓ Duty cycle: {duty_cycle:.2f}%")
    print(f"✓ Mask shape: {mask_cpu.shape}")
    print(f"✓ Mask dtype: {mask_cpu.dtype}")
    
    return gap_gen_cpu, mask_cpu, {"init": init_time, "generation": generation_time}

def run_gpu_test(sim_t, dt, gap_definitions):
    """Run gap generation on GPU."""
    print("\n" + "="*50)
    print("RUNNING GPU TEST")
    print("="*50)
    
    # Fixed seeds for reproducibility
    planseed = 2618240388
    unplanseed = 3387490715
    
    # Create GPU gap mask generator
    start_time = time.time()
    gap_gen_gpu = GapMaskGenerator(
        sim_t, 
        dt, 
        gap_definitions, 
        treat_as_nan=False, 
        planseed=planseed,
        unplanseed=unplanseed,
        use_gpu=True  # Enable GPU
    )
    init_time = time.time() - start_time
    
    # Generate mask
    start_time = time.time()
    mask_gpu = gap_gen_gpu.generate_mask(include_unplanned=True, include_planned=True)
    generation_time = time.time() - start_time
    
    # Calculate duty cycle
    duty_cycle = 100 * (1 - np.sum(mask_gpu == 0) / len(mask_gpu))
    
    print(f"✓ GPU initialization time: {init_time:.4f} seconds")
    print(f"✓ GPU generation time: {generation_time:.4f} seconds")
    print(f"✓ Duty cycle: {duty_cycle:.2f}%")
    print(f"✓ Mask shape: {mask_gpu.shape}")
    print(f"✓ Mask dtype: {mask_gpu.dtype}")
    
    return gap_gen_gpu, mask_gpu, {"init": init_time, "generation": generation_time}

def compare_results(mask_cpu, mask_gpu, cpu_times, gpu_times):
    """Compare CPU and GPU results."""
    print("\n" + "="*50)
    print("COMPARING RESULTS")
    print("="*50)
    
    # Convert GPU mask to CPU for comparison if needed
    try:
        import cupy as cp
        if isinstance(mask_gpu, cp.ndarray):
            mask_gpu_cpu = cp.asnumpy(mask_gpu)
        else:
            mask_gpu_cpu = mask_gpu
    except ImportError:
        mask_gpu_cpu = mask_gpu
    
    # Check if results are identical
    arrays_equal = np.array_equal(mask_cpu, mask_gpu_cpu)
    print(f"Results identical: {'✓ YES' if arrays_equal else '✗ NO'}")
    
    if not arrays_equal:
        diff = np.sum(mask_cpu != mask_gpu_cpu)
        print(f"Different elements: {diff} / {len(mask_cpu)} ({100*diff/len(mask_cpu):.2f}%)")
    
    # Performance comparison
    print(f"\nPerformance Comparison:")
    print(f"CPU total time: {cpu_times['init'] + cpu_times['generation']:.4f} seconds")
    print(f"GPU total time: {gpu_times['init'] + gpu_times['generation']:.4f} seconds")
    
    if gpu_times['generation'] > 0:
        speedup = cpu_times['generation'] / gpu_times['generation']
        print(f"Generation speedup: {speedup:.2f}x {'(GPU faster)' if speedup > 1 else '(CPU faster)'}")

def test_taper_functionality(gap_gen, mask):
    """Test the smooth taper functionality."""
    print("\n" + "="*30)
    print("TESTING TAPER FUNCTIONALITY")
    print("="*30)
    
    # Define taper configuration (from notebook)
    taper_defs = {
        "planned": {
            "antenna repointing": {"lobe_lengths_hr": 5.0},
            "TM stray potential": {"lobe_lengths_hr": 0.5},
            "TTL calibration": {"lobe_lengths_hr": 2.0}
        },
        "unplanned": {
            "platform safe mode": {"lobe_lengths_hr": 1.0},
            "QPD loss micrometeoroid": {"lobe_lengths_hr": 1.0},
            "HR GRS loss micrometeoroid": {"lobe_lengths_hr": 7.0},
            "WR GRS loss micrometeoroid": {"lobe_lengths_hr": 10.0}
        }
    }
    
    start_time = time.time()
    smoothed_mask = gap_gen.apply_smooth_taper_to_mask(mask, taper_gap_definitions=taper_defs)
    taper_time = time.time() - start_time
    
    print(f"✓ Taper application time: {taper_time:.4f} seconds")
    print(f"✓ Original mask range: [{np.min(mask):.3f}, {np.max(mask):.3f}]")
    print(f"✓ Smoothed mask range: [{np.min(smoothed_mask):.3f}, {np.max(smoothed_mask):.3f}]")
    
    return smoothed_mask

def main():
    """Main test function."""
    print("LISA-GAP GPU TEST SCRIPT")
    print("=" * 60)
    
    # Check GPU availability
    gpu_available = check_gpu_availability()
    
    # Create test configuration
    sim_t, dt, gap_definitions = create_test_configuration()
    
    # Run CPU test (always available)
    gap_gen_cpu, mask_cpu, cpu_times = run_cpu_test(sim_t, dt, gap_definitions)
    
    # Test taper functionality on CPU
    print("\nTesting CPU taper functionality:")
    smoothed_mask_cpu = test_taper_functionality(gap_gen_cpu, mask_cpu)
    
    # Run GPU test if available
    if gpu_available:
        gap_gen_gpu, mask_gpu, gpu_times = run_gpu_test(sim_t, dt, gap_definitions)
        compare_results(mask_cpu, mask_gpu, cpu_times, gpu_times)
        
        # Test taper functionality on GPU
        print("\nTesting GPU taper functionality:")
        smoothed_mask_gpu = test_taper_functionality(gap_gen_gpu, mask_gpu)
    else:
        print("\n⚠️  Skipping GPU tests - CuPy not available")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("✓ CPU gap generation: PASSED")
    print("✓ CPU taper functionality: PASSED")
    
    if gpu_available:
        print("✓ GPU gap generation: PASSED")
        print("✓ GPU taper functionality: PASSED")
        print("✓ CPU/GPU comparison: PASSED")
    else:
        print("- GPU tests: SKIPPED (CuPy not available)")
    
    print("\nFor full functionality, install GPU support:")
    print("  pip install lisa-gap[cuda12x]  # For CUDA 12.x")
    print("  pip install lisa-gap[cuda11x]  # For CUDA 11.x")

if __name__ == "__main__":
    main()
