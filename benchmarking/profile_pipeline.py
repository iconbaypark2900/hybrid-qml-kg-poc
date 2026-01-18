#!/usr/bin/env python3
"""Profile pipeline to identify bottlenecks"""

import sys
import os
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import cProfile
import pstats
import io
from scripts.run_pipeline import run_complete_pipeline

def main():
    parser = argparse.ArgumentParser(description="Profile pipeline execution")
    parser.add_argument("--output", type=str, default="results/profile_stats.txt",
                       help="Output file for profile stats")
    parser.add_argument("--sort", type=str, default="cumulative",
                       choices=["cumulative", "time", "calls"],
                       help="Sort key for profile output")
    parser.add_argument("--lines", type=int, default=50,
                       help="Number of lines to display")
    args = parser.parse_args()
    
    # Profile the pipeline
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        run_complete_pipeline()
    except Exception as e:
        print(f"Pipeline failed: {e}")
    
    profiler.disable()
    
    # Generate stats
    stats_stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stats_stream)
    stats.sort_stats(args.sort)
    stats.print_stats(args.lines)
    
    # Save to file
    with open(args.output, "w") as f:
        f.write(stats_stream.getvalue())
    
    print(f"\n✅ Profile saved to {args.output}")
    print("\nTop functions by cumulative time:")
    print(stats_stream.getvalue()[:2000])  # Print first 2000 chars

if __name__ == "__main__":
    main()

