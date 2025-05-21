#!/usr/bin/env python3
"""
Onshape to Stonefish Pipeline

Automates the complete pipeline from Onshape CAD models to Stonefish simulation scenarios:
1. Set environment variables from config file
2. Export URDF and meshes from Onshape using onshape-to-robot
3. Downsample meshes using pymeshlab with configurable limits
4. Convert URDF to Stonefish scenario format

Usage:
    python onshape_to_stonefish_pipeline.py [options]
"""

import argparse
import os
import sys
import yaml
import subprocess
import json
from pathlib import Path
from typing import Dict, Optional, List
import tempfile
import shutil
import pymeshlab

class OnshapeConfig:
    """Configuration for Onshape API access"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.api_url = "https://cad.onshape.com"
        self.access_key = ""
        self.secret_key = ""
        self.load_config()
    
    def load_config(self):
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            self.create_default_config()
            print(f"Created default config file at {self.config_path}")
            print("Please edit the config file with your Onshape API credentials")
            return
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        onshape_config = config.get('onshape', {})
        self.api_url = onshape_config.get('api_url', self.api_url)
        self.access_key = onshape_config.get('access_key', '')
        self.secret_key = onshape_config.get('secret_key', '')
    
    def create_default_config(self):
        """Create a default configuration file"""
        default_config = {
            'onshape': {
                'api_url': 'https://cad.onshape.com',
                'access_key': 'Your_Access_Key_Here',
                'secret_key': 'Your_Secret_Key_Here'
            },
            'mesh_downsampling': {
                'default_max_faces': 5000,
                'per_mesh_limits': {
                    'os_hull.stl': {
                        'max_faces': 10000
                    },
                    'thruster.stl': {
                        'max_faces': 1000
                    },
                    'os_arm_base.stl': {
                        'max_faces': 2000
                    },
                    'os_arm_link.stl': {
                        'max_faces': 1500
                    },
                    'os_arm_tube.stl': {
                        'max_faces': 1500
                    },
                    'os_launcher.stl': {
                        'max_faces': 3000
                    },
                    'os_dropper.stl': {
                        'max_faces': 2000
                    }
                }
            },
            'output': {
                'scenario_file': 'osprey.scn',
                'overwrite_existing': True
            }
        }
        
        # Create directory only if config_path contains a directory component
        config_dir = os.path.dirname(self.config_path)
        if config_dir:
            os.makedirs(config_dir, exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    def get_environment_dict(self) -> Dict[str, str]:
        """Get Onshape environment variables as a dictionary"""
        env_vars = {
            'ONSHAPE_API': self.api_url,
            'ONSHAPE_ACCESS_KEY': self.access_key,
            'ONSHAPE_SECRET_KEY': self.secret_key
        }
        
        print(f"Using Onshape environment variables:")
        print(f"  ONSHAPE_API = {self.api_url}")
        print(f"  ONSHAPE_ACCESS_KEY = {self.access_key[:8]}..." if self.access_key else "  ONSHAPE_ACCESS_KEY = (not set)")
        print(f"  ONSHAPE_SECRET_KEY = {self.secret_key[:8]}..." if self.secret_key else "  ONSHAPE_SECRET_KEY = (not set)")
        
        return env_vars

class MeshProcessor:
    """Handles mesh downsampling using pymeshlab"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.default_max_faces = config.get('default_max_faces', 5000)
        self.per_mesh_limits = config.get('per_mesh_limits', {})
    
    def downsample_mesh(self, mesh_path: str) -> bool:
        """Downsample a single mesh file"""
        try:
            # Load mesh
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(mesh_path)
            
            if ms.current_mesh().face_number() == 0:
                print(f"Warning: {mesh_path} has no faces, skipping downsampling")
                return True
            
            mesh_filename = os.path.basename(mesh_path)
            mesh_config = self.per_mesh_limits.get(mesh_filename, {})
            max_faces = mesh_config.get('max_faces', self.default_max_faces)
            
            print(f"Processing {mesh_filename}:")
            print(f"  Original: {ms.current_mesh().vertex_number()} vertices, {ms.current_mesh().face_number()} faces")
            print(f"  Target: ≤{max_faces} faces")
            
            # Check if downsampling is needed
            current_faces = ms.current_mesh().face_number()
            
            if max_faces < current_faces:
                # Downsample using quadric edge collapse decimation
                target_faces = min(max_faces, current_faces)
                ms.apply_filter('meshing_decimation_quadric_edge_collapse', 
                            targetfacenum=target_faces, 
                            preserveboundary=True, 
                            preservenormal=True,
                            # planarquadric=True,
                            preservetopology=True)
                
                
                # Compute normals
                ms.compute_normal_per_face()
                # ms.compute_normal_per_vertex()
                
                print(f"  Downsampled: {ms.current_mesh().vertex_number()} vertices, {ms.current_mesh().face_number()} faces")
            
            # Ensure triangulation
            # ms.apply_filter('meshing_repair_non_manifold_vertices')
            # ms.apply_filter('meshing_repair_non_manifold_edges')

            # Save as ASCII STL with normals
            ms.save_current_mesh(mesh_path, 
                                binary=False)  # ASCII format
            
            return True
            
        except Exception as e:
            print(f"Error processing {mesh_path}: {e}")
            return False
    
    def process_assets_directory(self, assets_dir: str) -> bool:
        """Process all STL files in the assets directory"""
        assets_path = Path(assets_dir)
        if not assets_path.exists():
            print(f"Assets directory {assets_dir} does not exist")
            return False
        
        stl_files = list(assets_path.glob("*.stl"))
        if not stl_files:
            print(f"No STL files found in {assets_dir}")
            return True
        
        print(f"\nDownsampling {len(stl_files)} STL files in {assets_dir}:")
        
        success = True
        for stl_file in stl_files:
            if not self.downsample_mesh(str(stl_file)):
                success = False
        
        return success

class OnshapePipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.onshape_config = OnshapeConfig(config_path)
        
        # Load full config for other components
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.full_config = yaml.safe_load(f)
        else:
            self.full_config = {}
        
        self.mesh_processor = MeshProcessor(self.full_config.get('mesh_downsampling', {}))
        self.output_config = self.full_config.get('output', {})
    
    def run_onshape_to_robot(self, osprey_dir: str, env_vars: Dict[str, str]) -> bool:
        """Run onshape-to-robot command with environment variables"""
        print(f"\nRunning onshape-to-robot on {osprey_dir}...")
        
        try:
            # Create environment with current environment plus Onshape variables
            env = os.environ.copy()
            env.update(env_vars)
            
            cmd = ['onshape-to-robot', osprey_dir]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
            print("onshape-to-robot completed successfully")
            if result.stdout:
                print("Output:", result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error running onshape-to-robot: {e}")
            if e.stdout:
                print("STDOUT:", e.stdout)
            if e.stderr:
                print("STDERR:", e.stderr)
            return False
        except FileNotFoundError:
            print("Error: onshape-to-robot command not found. Please install onshape-to-robot.")
            return False
    
    def run_urdf_to_stonefish(self, urdf_path: str, output_path: str, stonefish_config: str) -> bool:
        """Run urdf_to_stonefish conversion"""
        print(f"\nConverting URDF to Stonefish scenario...")
        
        script_dir = Path(__file__).parent
        urdf_converter = script_dir / "urdf_to_stonefish.py"
        
        if not urdf_converter.exists():
            print(f"Error: urdf_to_stonefish.py not found at {urdf_converter}")
            return False
        
        try:
            cmd = ['python3', str(urdf_converter), urdf_path, '-o', output_path, '-c', stonefish_config]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"URDF to Stonefish conversion completed: {output_path}")
            if result.stdout:
                print("Output:", result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error running urdf_to_stonefish: {e}")
            if e.stdout:
                print("STDOUT:", e.stdout)
            if e.stderr:
                print("STDERR:", e.stderr)
            return False
    
    def run_pipeline(self, osprey_dir: str) -> bool:
        """Run the complete pipeline"""
        osprey_path = Path(osprey_dir)
        
        if not osprey_path.exists():
            print(f"Error: Osprey directory {osprey_dir} does not exist")
            return False
        
        # Get environment variables for onshape-to-robot
        onshape_env = self.onshape_config.get_environment_dict()
        
        # Step 1: Run onshape-to-robot (unless skipped)
        if not self.run_onshape_to_robot(osprey_dir, onshape_env):
            return False
        
        # Step 2: Process meshes (unless skipped)
        assets_dir = osprey_path / "assets"
        if not self.mesh_processor.process_assets_directory(str(assets_dir)):
            print("Warning: Some mesh processing failed, continuing...")
        
        # Step 3: Convert URDF to Stonefish
        urdf_path = osprey_path / "robot.urdf"
        if not urdf_path.exists():
            print(f"Error: robot.urdf not found at {urdf_path}")
            return False
        
        # Determine output path
        output_filename = self.output_config.get('scenario_file', 'osprey.scn')
        if os.path.isabs(output_filename):
            output_path = output_filename
        else:
            # Place in scenarios directory relative to the script
            script_dir = Path(__file__).parent.parent
            scenarios_dir = script_dir / "scenarios"
            output_path = scenarios_dir / output_filename
        
        # Check if output exists and if we should overwrite
        if output_path.exists() and not self.output_config.get('overwrite_existing', True):
            print(f"Error: Output file {output_path} already exists and overwrite is disabled")
            return False
        
        stonefish_config = Path(__file__).parent / "stonefish_config.yaml"
        if not self.run_urdf_to_stonefish(str(urdf_path), str(output_path), str(stonefish_config)):
            return False
        
        print(f"\nPipeline completed successfully!")
        print(f"Generated scenario file: {output_path}")
        return True

def main():
    parser = argparse.ArgumentParser(description='Onshape to Stonefish conversion pipeline')
    
    args = parser.parse_args()
    
    # Default to relative path from script location
    script_dir = Path(__file__).parent.parent
    osprey_dir = script_dir / "data" / "osprey"
    config_path = Path(__file__).parent / "onshape_config.yaml"
    
    # Run pipeline
    pipeline = OnshapePipeline(str(config_path))
    if not pipeline.run_pipeline(str(osprey_dir)):
        sys.exit(1)

if __name__ == '__main__':
    main() 