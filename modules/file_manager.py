"""
Safe File Manager Module for Turkish AI Terminal Assistant

This module provides safe file management operations with validation,
error handling, and security checks to prevent unauthorized access
and unintended file modifications.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


class FileManager:
    """
    Safe file manager for handling file operations with security checks
    and validation.
    """

    def __init__(self, base_path: Optional[str] = None, max_file_size: int = 100 * 1024 * 1024):
        """
        Initialize FileManager with optional base path and size restrictions.

        Args:
            base_path: Base directory for operations (defaults to current directory)
            max_file_size: Maximum file size in bytes (default: 100MB)
        """
        self.base_path = Path(base_path or os.getcwd())
        self.max_file_size = max_file_size

        if not self.base_path.exists():
            raise ValueError(f"Base path does not exist: {self.base_path}")

        logger.info(f"FileManager initialized with base path: {self.base_path}")

    def _validate_path(self, file_path: Union[str, Path]) -> Path:
        """
        Validate and resolve file path to prevent directory traversal attacks.

        Args:
            file_path: Path to validate

        Returns:
            Resolved Path object

        Raises:
            ValueError: If path is invalid or outside base directory
        """
        path = Path(file_path)

        # Make path absolute relative to base_path if it's relative
        if not path.is_absolute():
            path = self.base_path / path

        # Resolve to handle .. and symlinks
        try:
            resolved_path = path.resolve()
        except Exception as e:
            raise ValueError(f"Failed to resolve path: {e}")

        # Check if path is within base_path
        try:
            resolved_path.relative_to(self.base_path.resolve())
        except ValueError:
            raise ValueError(f"Path outside base directory: {file_path}")

        return resolved_path

    def read_file(self, file_path: Union[str, Path], encoding: str = "utf-8") -> str:
        """
        Safely read file contents.

        Args:
            file_path: Path to file to read
            encoding: File encoding (default: utf-8)

        Returns:
            File contents as string

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If path is invalid
            IOError: If read operation fails
        """
        try:
            validated_path = self._validate_path(file_path)

            if not validated_path.is_file():
                raise FileNotFoundError(f"File not found: {file_path}")

            if validated_path.stat().st_size > self.max_file_size:
                raise ValueError(
                    f"File too large: {validated_path.stat().st_size} bytes "
                    f"(max: {self.max_file_size} bytes)"
                )

            with open(validated_path, "r", encoding=encoding) as f:
                content = f.read()

            logger.info(f"Successfully read file: {file_path}")
            return content

        except (FileNotFoundError, ValueError, IOError) as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise

    def write_file(
        self,
        file_path: Union[str, Path],
        content: str,
        encoding: str = "utf-8",
        overwrite: bool = False
    ) -> Path:
        """
        Safely write content to file.

        Args:
            file_path: Path where to write file
            content: Content to write
            encoding: File encoding (default: utf-8)
            overwrite: Whether to overwrite existing file (default: False)

        Returns:
            Path to written file

        Raises:
            ValueError: If path is invalid or file exists and overwrite is False
            IOError: If write operation fails
        """
        try:
            validated_path = self._validate_path(file_path)

            if validated_path.exists() and not overwrite:
                raise ValueError(f"File already exists: {file_path}. Set overwrite=True to replace.")

            # Create parent directories if they don't exist
            validated_path.parent.mkdir(parents=True, exist_ok=True)

            # Check content size
            if len(content.encode(encoding)) > self.max_file_size:
                raise ValueError(
                    f"Content too large: {len(content.encode(encoding))} bytes "
                    f"(max: {self.max_file_size} bytes)"
                )

            with open(validated_path, "w", encoding=encoding) as f:
                f.write(content)

            logger.info(f"Successfully wrote file: {file_path}")
            return validated_path

        except (ValueError, IOError) as e:
            logger.error(f"Error writing file {file_path}: {e}")
            raise

    def append_file(self, file_path: Union[str, Path], content: str, encoding: str = "utf-8") -> Path:
        """
        Safely append content to existing file.

        Args:
            file_path: Path to file
            content: Content to append
            encoding: File encoding (default: utf-8)

        Returns:
            Path to modified file

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If path is invalid
            IOError: If operation fails
        """
        try:
            validated_path = self._validate_path(file_path)

            if not validated_path.is_file():
                raise FileNotFoundError(f"File not found: {file_path}")

            with open(validated_path, "a", encoding=encoding) as f:
                f.write(content)

            logger.info(f"Successfully appended to file: {file_path}")
            return validated_path

        except (FileNotFoundError, ValueError, IOError) as e:
            logger.error(f"Error appending to file {file_path}: {e}")
            raise

    def delete_file(self, file_path: Union[str, Path]) -> bool:
        """
        Safely delete a file.

        Args:
            file_path: Path to file to delete

        Returns:
            True if file was deleted, False if file didn't exist

        Raises:
            ValueError: If path is invalid
            IOError: If deletion fails
        """
        try:
            validated_path = self._validate_path(file_path)

            if validated_path.is_file():
                validated_path.unlink()
                logger.info(f"Successfully deleted file: {file_path}")
                return True
            else:
                logger.warning(f"File not found for deletion: {file_path}")
                return False

        except (ValueError, IOError) as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            raise

    def create_directory(self, dir_path: Union[str, Path]) -> Path:
        """
        Safely create directory with parent directories if needed.

        Args:
            dir_path: Path to directory to create

        Returns:
            Path to created directory

        Raises:
            ValueError: If path is invalid
            IOError: If creation fails
        """
        try:
            validated_path = self._validate_path(dir_path)
            validated_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Successfully created directory: {dir_path}")
            return validated_path

        except (ValueError, IOError) as e:
            logger.error(f"Error creating directory {dir_path}: {e}")
            raise

    def delete_directory(self, dir_path: Union[str, Path], recursive: bool = False) -> bool:
        """
        Safely delete a directory.

        Args:
            dir_path: Path to directory to delete
            recursive: Whether to recursively delete contents (default: False)

        Returns:
            True if directory was deleted, False if it didn't exist

        Raises:
            ValueError: If path is invalid
            IOError: If directory is not empty and recursive is False
        """
        try:
            validated_path = self._validate_path(dir_path)

            if validated_path.is_dir():
                if recursive:
                    shutil.rmtree(validated_path)
                    logger.info(f"Successfully deleted directory recursively: {dir_path}")
                else:
                    validated_path.rmdir()
                    logger.info(f"Successfully deleted directory: {dir_path}")
                return True
            else:
                logger.warning(f"Directory not found for deletion: {dir_path}")
                return False

        except (ValueError, IOError) as e:
            logger.error(f"Error deleting directory {dir_path}: {e}")
            raise

    def list_files(self, dir_path: Union[str, Path] = ".", recursive: bool = False) -> List[Dict[str, Any]]:
        """
        List files in directory with metadata.

        Args:
            dir_path: Directory path to list (default: current base path)
            recursive: Whether to list recursively (default: False)

        Returns:
            List of file information dictionaries

        Raises:
            ValueError: If path is invalid
            IOError: If listing fails
        """
        try:
            if dir_path == ".":
                validated_path = self.base_path
            else:
                validated_path = self._validate_path(dir_path)

            if not validated_path.is_dir():
                raise ValueError(f"Not a directory: {dir_path}")

            files_info = []
            pattern = "**/*" if recursive else "*"

            for file_path in validated_path.glob(pattern):
                if file_path.is_file():
                    stat_info = file_path.stat()
                    files_info.append({
                        "path": str(file_path.relative_to(self.base_path)),
                        "size": stat_info.st_size,
                        "modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                        "is_file": True
                    })

            logger.info(f"Listed {len(files_info)} files in {dir_path}")
            return files_info

        except (ValueError, IOError) as e:
            logger.error(f"Error listing directory {dir_path}: {e}")
            raise

    def copy_file(self, source: Union[str, Path], destination: Union[str, Path]) -> Path:
        """
        Safely copy a file.

        Args:
            source: Source file path
            destination: Destination file path

        Returns:
            Path to copied file

        Raises:
            FileNotFoundError: If source doesn't exist
            ValueError: If paths are invalid
            IOError: If copy fails
        """
        try:
            source_path = self._validate_path(source)
            dest_path = self._validate_path(destination)

            if not source_path.is_file():
                raise FileNotFoundError(f"Source file not found: {source}")

            # Create parent directories if needed
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(source_path, dest_path)
            logger.info(f"Successfully copied file from {source} to {destination}")
            return dest_path

        except (FileNotFoundError, ValueError, IOError) as e:
            logger.error(f"Error copying file: {e}")
            raise

    def move_file(self, source: Union[str, Path], destination: Union[str, Path]) -> Path:
        """
        Safely move a file.

        Args:
            source: Source file path
            destination: Destination file path

        Returns:
            Path to moved file

        Raises:
            FileNotFoundError: If source doesn't exist
            ValueError: If paths are invalid
            IOError: If move fails
        """
        try:
            source_path = self._validate_path(source)
            dest_path = self._validate_path(destination)

            if not source_path.is_file():
                raise FileNotFoundError(f"Source file not found: {source}")

            # Create parent directories if needed
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.move(str(source_path), str(dest_path))
            logger.info(f"Successfully moved file from {source} to {destination}")
            return dest_path

        except (FileNotFoundError, ValueError, IOError) as e:
            logger.error(f"Error moving file: {e}")
            raise

    def file_exists(self, file_path: Union[str, Path]) -> bool:
        """
        Check if file exists safely.

        Args:
            file_path: Path to check

        Returns:
            True if file exists, False otherwise
        """
        try:
            validated_path = self._validate_path(file_path)
            return validated_path.is_file()
        except ValueError:
            return False

    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get detailed file information.

        Args:
            file_path: Path to file

        Returns:
            Dictionary with file information

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If path is invalid
        """
        try:
            validated_path = self._validate_path(file_path)

            if not validated_path.is_file():
                raise FileNotFoundError(f"File not found: {file_path}")

            stat_info = validated_path.stat()
            return {
                "path": str(validated_path),
                "relative_path": str(validated_path.relative_to(self.base_path)),
                "size": stat_info.st_size,
                "created": datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                "accessed": datetime.fromtimestamp(stat_info.st_atime).isoformat(),
                "is_file": True
            }

        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            raise


# Convenience functions for simple operations
def read_file(file_path: Union[str, Path], base_path: Optional[str] = None) -> str:
    """Convenience function to read file."""
    manager = FileManager(base_path)
    return manager.read_file(file_path)


def write_file(
    file_path: Union[str, Path],
    content: str,
    base_path: Optional[str] = None,
    overwrite: bool = False
) -> Path:
    """Convenience function to write file."""
    manager = FileManager(base_path)
    return manager.write_file(file_path, content, overwrite=overwrite)


def delete_file(file_path: Union[str, Path], base_path: Optional[str] = None) -> bool:
    """Convenience function to delete file."""
    manager = FileManager(base_path)
    return manager.delete_file(file_path)
