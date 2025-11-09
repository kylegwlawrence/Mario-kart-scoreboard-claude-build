#!/usr/bin/env python3
"""
Generate pipeline configuration files with all valid preprocessing chain combinations.

This script creates preprocessing chains that satisfy the following constraints:
1. Every chain starts with: grayscale -> inversion
2. Edge detection can only appear after grayscale+inversion
3. Dilate and erode must be adjacent with the same kernel size
4. Dilation/erosion cannot precede edge_detection
5. For edge_detection: hysteresis_max > hysteresis_min (strict inequality)

Usage:
    python3 generate_pipeline_config.py --output easyocr_generated.json \
        --max-chains 100 --include threshold gaussian_blur
"""

import json
import argparse
import itertools
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from utils import setup_logger


class PipelineConfigGenerator:
    """Generates valid preprocessing pipeline configurations."""

    def __init__(self, include_methods: Optional[List[str]] = None,
                 exclude_methods: Optional[List[str]] = None):
        """
        Initialize generator with optional method filters.

        Args:
            include_methods: Only use these methods (None = all)
            exclude_methods: Exclude these methods
        """
        self.include_methods = include_methods
        self.exclude_methods = exclude_methods or []

        # Define parameter variations for each method
        self.method_params = {
            'gaussian_blur': [
                {'kernel': [k, k], 'sigmaX': 0, 'sigmaY': 0}
                for k in [3, 5, 7, 9, 11]
            ],
            'edge_detection': self._generate_edge_detection_params(),
            'dilate': [
                {'kernel': [k, k]}
                for k in [3, 5, 7, 9, 11]
            ],
            'erode': [
                {'kernel': [k, k]}
                for k in [3, 5, 7, 9, 11]
            ],
            'threshold': [
                {'threshold': t, 'max_value': 255}
                for t in [100, 127, 150]
            ],
            'adaptive_threshold': [
                {'max_value': 255, 'block_size': b, 'C': c}
                for b in [11, 15, 21]
                for c in [2, -2]
            ],
            'blur': [
                {'kernel': [k, k]}
                for k in [3, 5]
            ],
            'median_blur': [
                {'ksize': k}
                for k in [3, 5, 7]
            ],
            'bilateral_filter': [
                {'d': 9, 'sigmaColor': 75, 'sigmaSpace': 75}
            ],
            'morphology': [
                {'operation': op, 'kernel': [k, k]}
                for op in ['open', 'close']
                for k in [3, 5, 7, 9, 11]
            ],
            'contrast': [
                {'alpha': a, 'beta': b}
                for a in [1.0, 1.2]
                for b in [0, 10]
            ],
            'downscale': [
                {'scale_factor': s}
                for s in [0.5, 0.75]
            ],
            # Methods with no parameters
            'grayscale': [None],
            'inversion': [None],
        }

    @staticmethod
    def _generate_edge_detection_params() -> List[Dict[str, int]]:
        """Generate edge detection parameter combinations with hysteresis_max > hysteresis_min."""
        params = []
        hysteresis_mins = [100, 125, 150]
        hysteresis_maxs = [150, 175, 200]

        for min_val in hysteresis_mins:
            for max_val in hysteresis_maxs:
                if max_val > min_val:  # Enforce constraint
                    params.append({
                        'hysteresis_min': min_val,
                        'hysteresis_max': max_val
                    })

        return params

    def _filter_methods(self) -> set:
        """Get set of methods to use based on include/exclude filters."""
        all_methods = set(self.method_params.keys())

        if self.include_methods:
            all_methods = all_methods & set(self.include_methods)

        all_methods -= set(self.exclude_methods)
        return all_methods

    def _generate_base_chain(self) -> List[Dict[str, Any]]:
        """Generate the mandatory base chain: grayscale -> inversion."""
        return [
            {'method': 'grayscale', 'parameters': None},
            {'method': 'inversion', 'parameters': None},
        ]

    def _generate_morphology_pairs(self) -> List[Tuple[Dict, Dict]]:
        """Generate valid dilate+erode pairs (adjacent with same kernel)."""
        pairs = []
        kernels = [3, 5, 7, 9]

        for k in kernels:
            dilate_config = {'method': 'dilate', 'parameters': {'kernel': [k, k]}}
            erode_config = {'method': 'erode', 'parameters': {'kernel': [k, k]}}
            pairs.append((dilate_config, erode_config))
            # Also allow erode then dilate
            pairs.append((erode_config, dilate_config))

        return pairs

    def _has_edge_detection(self, chain: List[Dict]) -> bool:
        """Check if chain contains edge_detection."""
        return any(m['method'] == 'edge_detection' for m in chain)

    def _can_add_morphology(self, chain: List[Dict]) -> bool:
        """Check if morphology operations can be added (must be after edge detection)."""
        return self._has_edge_detection(chain)

    def generate_chains(self, max_chains: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generate all valid preprocessing chains.

        Args:
            max_chains: Maximum number of chains to generate (None = unlimited)

        Returns:
            List of preprocessing chain configurations
        """
        chains = []
        available_methods = self._filter_methods()

        # Rule 1: Generate basic chains with only grayscale and inversion (both orders)
        chains.append([
            {'method': 'grayscale', 'parameters': None},
            {'method': 'inversion', 'parameters': None},
        ])
        if max_chains and len(chains) >= max_chains:
            return chains

        chains.append([
            {'method': 'inversion', 'parameters': None},
            {'method': 'grayscale', 'parameters': None},
        ])
        if max_chains and len(chains) >= max_chains:
            return chains

        # Rule 2: Both basic combinations with edge detection only
        edge_params_list = self.method_params.get('edge_detection', [])
        for edge_params in edge_params_list:
            chain = [
                {'method': 'grayscale', 'parameters': None},
                {'method': 'inversion', 'parameters': None},
                {'method': 'edge_detection', 'parameters': edge_params},
            ]
            chains.append(chain)
            if max_chains and len(chains) >= max_chains:
                return chains

            chain = [
                {'method': 'inversion', 'parameters': None},
                {'method': 'grayscale', 'parameters': None},
                {'method': 'edge_detection', 'parameters': edge_params},
            ]
            chains.append(chain)
            if max_chains and len(chains) >= max_chains:
                return chains

        # Rule 3: Generate all other combinations starting with standard base chain
        base_chain = self._generate_base_chain()

        # Generate single-method additions (after base)
        for method in available_methods:
            if method in ['grayscale', 'inversion']:
                continue  # Already in base

            for params in self.method_params[method]:
                chain = base_chain.copy()
                chain.append({'method': method, 'parameters': params})
                chains.append(chain)

                if max_chains and len(chains) >= max_chains:
                    return chains

        # Generate two-method combinations
        single_methods = [m for m in available_methods if m not in ['grayscale', 'inversion']]

        for method1, method2 in itertools.combinations(single_methods, 2):
            for params1 in self.method_params[method1]:
                for params2 in self.method_params[method2]:
                    # Check morphology constraint
                    if (method1 in ['dilate', 'erode'] or method2 in ['dilate', 'erode']):
                        continue  # Skip individual dilate/erode, use pairs instead

                    chain = base_chain.copy()
                    chain.append({'method': method1, 'parameters': params1})
                    chain.append({'method': method2, 'parameters': params2})
                    chains.append(chain)

                    if max_chains and len(chains) >= max_chains:
                        return chains

        # Generate morphology pair chains (dilate+erode adjacent with same kernel)
        morphology_pairs = self._generate_morphology_pairs()
        for pair in morphology_pairs:
            chain = base_chain.copy()
            chain.extend(pair)
            chains.append(chain)

            if max_chains and len(chains) >= max_chains:
                return chains

        # Generate chains with edge detection + other methods
        edge_params_list = self.method_params.get('edge_detection', [])
        other_methods = [m for m in single_methods if m != 'edge_detection']

        for edge_params in edge_params_list:
            # Edge detection alone
            chain = base_chain.copy()
            chain.append({'method': 'edge_detection', 'parameters': edge_params})
            chains.append(chain)

            if max_chains and len(chains) >= max_chains:
                return chains

            # Edge detection + other methods + morphology
            for method in other_methods:
                if method in ['dilate', 'erode']:
                    continue

                for params in self.method_params[method]:
                    chain = base_chain.copy()
                    chain.append({'method': method, 'parameters': params})
                    chain.append({'method': 'edge_detection', 'parameters': edge_params})
                    chains.append(chain)

                    if max_chains and len(chains) >= max_chains:
                        return chains

            # Edge detection + morphology pairs
            for pair in morphology_pairs:
                chain = base_chain.copy()
                chain.append({'method': 'edge_detection', 'parameters': edge_params})
                chain.extend(pair)
                chains.append(chain)

                if max_chains and len(chains) >= max_chains:
                    return chains

        return chains

    def generate_config(self, max_chains: Optional[int] = None,
                       image_source: str = "./pngs",
                       primary_engine: str = "easyocr",
                       fuzzy_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Generate complete pipeline configuration.

        Args:
            max_chains: Maximum number of chains to generate
            image_source: Path to image source directory
            primary_engine: OCR engine to use
            fuzzy_threshold: Minimum similarity score for player name fuzzy matching (0-1)

        Returns:
            Complete pipeline config dict
        """
        chains = self.generate_chains(max_chains)

        return {
            'image_source': image_source,
            'output_paths': {
                'annotated': 'output/annotated_images',
                'predictions': 'output/predictions',
                'cell_images': 'output/cell_images',
                'logs': '.logging'
            },
            'primary_engine': primary_engine,
            'retry_attempts': len(chains),
            'fuzzy_threshold': fuzzy_threshold,
            'preprocessing_chains': [
                {
                    'retry_attempt': idx,
                    'methods': chain
                }
                for idx, chain in enumerate(chains)
            ]
        }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Generate pipeline configuration with all valid preprocessing combinations'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output JSON filename (relative to project root)'
    )
    parser.add_argument(
        '--max-chains',
        type=int,
        default=None,
        help='Maximum number of chains to generate (default: unlimited)'
    )
    parser.add_argument(
        '--include',
        nargs='+',
        help='Only include these preprocessing methods'
    )
    parser.add_argument(
        '--exclude',
        nargs='+',
        help='Exclude these preprocessing methods'
    )
    parser.add_argument(
        '--image-source',
        default='./pngs',
        help='Image source path (default: ./pngs)'
    )
    parser.add_argument(
        '--engine',
        default='easyocr',
        help='OCR engine (default: easyocr)'
    )
    parser.add_argument(
        '--fuzzy-threshold',
        type=float,
        default=0.8,
        help='Fuzzy matching threshold for player names (0-1, default: 0.8)'
    )

    args = parser.parse_args()

    # Set up logging
    logger = setup_logger(
        name='PipelineConfigGenerator',
        log_dir='.logging',
        debug=False,
        console_output=True
    )

    # Generate config
    generator = PipelineConfigGenerator(
        include_methods=args.include,
        exclude_methods=args.exclude
    )

    config = generator.generate_config(
        max_chains=args.max_chains,
        image_source=args.image_source,
        primary_engine=args.engine,
        fuzzy_threshold=args.fuzzy_threshold
    )

    # Write output
    output_path = Path('src/configs/pipelines') / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Written pipeline config with {len(config['preprocessing_chains'])} chains to: {output_path}")
    print(f"Generated pipeline config with {len(config['preprocessing_chains'])} chains")
    print(f"Saved to: {output_path}")


if __name__ == '__main__':
    main()
