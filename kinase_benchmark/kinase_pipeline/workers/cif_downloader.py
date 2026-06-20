"""
CIF file downloader for PDB structures
"""

import os
import requests
import gzip
from pathlib import Path
from typing import List, Dict, Optional
import time
import logging


class CIFDownloader:
    """Downloads CIF files from PDB"""
    
    def __init__(self, download_dir: str = "cif_files", logger=None):
        """
        Initialize the downloader
        
        Args:
            download_dir: Directory to store downloaded CIF files
            logger: Logger instance for logging messages
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        
        self.pdb_base_url = "https://files.rcsb.org/download/"
        self.logger = logger
        
    def download_single(self, pdb_id: str, use_mmcif: bool = True) -> Optional[str]:
        """
        Download a single CIF file
        
        Args:
            pdb_id: PDB ID to download
            use_mmcif: Whether to use mmCIF format (recommended)
            
        Returns:
            Path to downloaded file, or None if failed
        """
        pdb_id = pdb_id.upper()
    
        if use_mmcif:
            url = f"{self.pdb_base_url}{pdb_id}.cif"
            filename = f"{pdb_id}.cif"
        else: # Fallback to PDB format
            url = f"{self.pdb_base_url}{pdb_id}.pdb"
            filename = f"{pdb_id}.pdb"
    
        file_path = self.download_dir / filename
        
        # Skip if already exists
        if file_path.exists():
            if self.logger: self.logger.debug(f"  File {filename} already exists, skipping")
            return str(file_path)
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        
            # Write the file
            with open(file_path, 'w') as f:
                f.write(response.text)
            
            if self.logger: self.logger.info(f"  ✓ Downloaded {filename}")
            return str(file_path)
            
        except requests.exceptions.RequestException as e:
            if self.logger: self.logger.error(f"  ✗ Failed to download {pdb_id}: {e}")
            return None
    
    def download_batch(self, pdb_ids: List[str], delay: float = 0.1) -> Dict[str, str]:
        """
        Download multiple CIF files
        
        Args:
            pdb_ids: List of PDB IDs to download
            delay: Delay between downloads (seconds)
            
        Returns:
            Dictionary mapping PDB ID to file path for successful downloads
        """
        downloaded_files = {}
        
        if self.logger: self.logger.info(f"Downloading {len(pdb_ids)} CIF files...")
        
        for i, pdb_id in enumerate(pdb_ids):
            if self.logger: self.logger.info(f"Downloading {i+1}/{len(pdb_ids)}: {pdb_id}")
            
            file_path = self.download_single(pdb_id)
            if file_path:
                downloaded_files[pdb_id.upper()] = file_path
            
            # Add delay to be respectful to the server
            if delay > 0 and i < len(pdb_ids) - 1:
                time.sleep(delay)
        
        if self.logger: self.logger.info(f"Successfully downloaded {len(downloaded_files)}/{len(pdb_ids)} files")
        return downloaded_files
 
    def get_file_path(self, pdb_id: str) -> Optional[str]:
        """
        Get the path to a downloaded CIF file
        
        Args:
            pdb_id: PDB ID
            
        Returns:
            Path to file if it exists, None otherwise
        """
        file_path = self.download_dir / f"{pdb_id.upper()}.cif"
        return str(file_path) if file_path.exists() else None
    
    def list_downloaded_files(self) -> List[str]:
        """
        List all downloaded CIF files
        
        Returns:
            List of PDB IDs for downloaded files
        """
        pdb_ids = []
        for file_path in self.download_dir.glob("*.cif"):
            pdb_id = file_path.stem.upper()
            if len(pdb_id) == 4:  # Valid PDB ID length
                pdb_ids.append(pdb_id)
        
        return sorted(pdb_ids)
