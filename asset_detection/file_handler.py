import os
import tempfile
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageOps


class FileHandler:
    """Handles file storage and temporary file operations"""

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=10)

    @staticmethod
    def save_crop(image_bytes, storage_path, filename):
        """Save cropped image to local storage and return path"""
        try:
            # Create directory if it doesn't exist
            Path(storage_path).mkdir(parents=True, exist_ok=True)

            # Save file locally
            full_path = Path(storage_path) / filename
            with open(full_path, 'wb') as f:
                f.write(image_bytes)

            print(f"DEBUG: Saved file to: {full_path}")
            return {
                'path': str(full_path),
                'url': str(full_path)  # For local files, URL is just the path
            }
        except Exception as e:
            print(f"Error saving crop: {e}")
            return None

    async def save_crop_async(self, image_bytes, storage_path, filename):
        """Save cropped image to local storage (asynchronous)"""
        loop = asyncio.get_event_loop()
        try:
            # Run the blocking file I/O in thread pool
            result = await loop.run_in_executor(
                self._executor,
                self.save_crop,
                image_bytes,
                storage_path,
                filename
            )
            return result
        except Exception as e:
            print(f"Error in async save_crop: {e}")
            return None

    @staticmethod
    def create_temp_file(uploaded_file, suffix='.jpg'):
        """
        Create a temporary file from Django uploaded file.
        Converts HEIC/HEIF images to JPEG for compatibility with OpenCV.
        Handles EXIF orientation data.
        """
        # Reset file pointer to beginning
        uploaded_file.seek(0)
        
        try:
            # Open image with PIL to handle HEIC/HEIF and EXIF orientation
            img = Image.open(uploaded_file)
            
            # Handle EXIF orientation (important for iPhone images)
            # ImageOps.exif_transpose() automatically applies EXIF orientation
            try:
                img = ImageOps.exif_transpose(img)
            except Exception:
                # If exif_transpose fails, try manual rotation (fallback)
                try:
                    exif = img.getexif()
                    if exif is not None:
                        orientation = exif.get(274)  # 274 is EXIF Orientation tag
                        if orientation == 3:
                            img = img.rotate(180, expand=True)
                        elif orientation == 6:
                            img = img.rotate(270, expand=True)
                        elif orientation == 8:
                            img = img.rotate(90, expand=True)
                except (AttributeError, KeyError, TypeError):
                    # No EXIF data or error reading it, continue
                    pass
            
            # Convert to RGB if necessary (handles RGBA, P mode, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save as JPEG to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            img.save(temp_file, format='JPEG', quality=95)
            temp_file.close()
            
            return temp_file.name
            
        except Exception as e:
            # Fallback to original method if PIL processing fails
            print(f"Warning: PIL image processing failed ({e}), using direct file copy")
            uploaded_file.seek(0)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            for chunk in uploaded_file.chunks():
                temp_file.write(chunk)
            temp_file.close()
            return temp_file.name

    @staticmethod
    def cleanup_temp_file(file_path):
        """Remove temporary file"""
        try:
            os.unlink(file_path)
        except OSError:
            pass

    @staticmethod
    def cleanup_storage_path(storage_path):
        """Delete all files in a given storage path"""
        try:
            # List all files in the directory
            path_obj = Path(storage_path)
            if not path_obj.exists():
                return 0

            files = [f for f in path_obj.iterdir() if f.is_file()]

            # Delete all files
            for file_path in files:
                try:
                    file_path.unlink()
                    print(f"DEBUG: Deleted old file: {file_path}")
                except Exception as e:
                    print(f"Warning: Could not delete {file_path}: {e}")

            print(f"DEBUG: Cleaned up {len(files)} files from {storage_path}")
            return len(files)
        except FileNotFoundError:
            # Directory doesn't exist, which is fine
            print(f"DEBUG: Storage path {storage_path} does not exist, no cleanup needed")
            return 0
        except Exception as e:
            print(f"Warning: Error during storage cleanup: {e}")
            return 0
