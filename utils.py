"""
Utility functions for the Library Human Detection System
"""
import json
import os
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import colorama
from colorama import Fore, Style

colorama.init(autoreset=True)


class ConfigManager:
    """Manages configuration loading and validation"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if not os.path.exists(self.config_path):
            print(f"{Fore.YELLOW}⚠ Config file not found. Creating default config...{Style.RESET_ALL}")
            self.create_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            print(f"{Fore.GREEN}✓ Configuration loaded successfully{Style.RESET_ALL}")
            return config
        except json.JSONDecodeError as e:
            print(f"{Fore.RED}❌ Error parsing config file: {e}{Style.RESET_ALL}")
            raise
        except Exception as e:
            print(f"{Fore.RED}❌ Error loading config: {e}{Style.RESET_ALL}")
            raise
    
    def create_default_config(self):
        """Create default configuration file"""
        default_config = {
            "camera": {"index": 0, "width": 1280, "height": 720, "fps": 30},
            "detection": {
                "model_path": "pose_landmarker_lite.task",
                "capture_cooldown": 5
            },
            "email": {
                "sender_email": "",
                "sender_password": "",
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587
            },
            "storage": {
                "save_folder": "./captured_images",
                "image_format": "jpg"
            }
        }
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'email.sender_email')"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
    
    def save(self):
        """Save configuration to file"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)


class Logger:
    """Custom logger with file and console output"""
    
    def __init__(self, name: str, log_folder: str = "logs", log_level: str = "INFO"):
        self.log_folder = log_folder
        os.makedirs(log_folder, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # File handler
        log_file = os.path.join(log_folder, f"{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Formatter - Use ASCII-safe encoding for Windows console
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Set encoding to UTF-8 for file handler to avoid Windows encoding issues
        try:
            file_handler.stream.reconfigure(encoding='utf-8')
        except (AttributeError, ValueError):
            # Fallback if reconfigure not available
            pass
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def debug(self, message: str):
        self.logger.debug(message)


class Statistics:
    """Track system statistics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.detection_count = 0
        self.image_capture_count = 0
        self.email_sent_count = 0
        self.email_failed_count = 0
        self.start_time = datetime.now()
        self.last_detection_time = None
        self.fps_history = []
        self.frame_count = 0
    
    def increment_detection(self):
        self.detection_count += 1
        self.last_detection_time = datetime.now()
    
    def increment_capture(self):
        self.image_capture_count += 1
    
    def increment_email_sent(self):
        self.email_sent_count += 1
    
    def increment_email_failed(self):
        self.email_failed_count += 1
    
    def update_fps(self, fps: float):
        self.fps_history.append(fps)
        if len(self.fps_history) > 100:  # Keep last 100 FPS readings
            self.fps_history.pop(0)
    
    def increment_frame(self):
        self.frame_count += 1
    
    def get_average_fps(self) -> float:
        if not self.fps_history:
            return 0.0
        return sum(self.fps_history) / len(self.fps_history)
    
    def get_uptime(self) -> str:
        uptime = datetime.now() - self.start_time
        hours, remainder = divmod(int(uptime.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            "detections": self.detection_count,
            "images_captured": self.image_capture_count,
            "emails_sent": self.email_sent_count,
            "emails_failed": self.email_failed_count,
            "uptime": self.get_uptime(),
            "average_fps": round(self.get_average_fps(), 2),
            "total_frames": self.frame_count
        }


def create_save_folder(folder_path: str) -> str:
    """Create save folder if it doesn't exist"""
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def format_timestamp() -> str:
    """Get formatted timestamp string"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_banner():
    """Print welcome banner"""
    banner = f"""
{Fore.CYAN}{'='*70}
{Fore.CYAN}    LIBRARY HUMAN DETECTION SYSTEM
{Fore.CYAN}    Production Version 2.0
{Fore.CYAN}{'='*70}{Style.RESET_ALL}
"""
    print(banner)


def print_success(message: str):
    """Print success message"""
    print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")


def print_error(message: str):
    """Print error message"""
    print(f"{Fore.RED}❌ {message}{Style.RESET_ALL}")


def print_warning(message: str):
    """Print warning message"""
    print(f"{Fore.YELLOW}⚠ {message}{Style.RESET_ALL}")


def print_info(message: str):
    """Print info message"""
    print(f"{Fore.BLUE}ℹ {message}{Style.RESET_ALL}")


class LocationService:
    """Get current GPS location"""
    
    def __init__(self):
        self.latitude = None
        self.longitude = None
        self.location_name = None
        self.city = None
        self.region = None
        self.country = None
        self.last_update = None
        self.update_interval = 30  # Update location every 30 seconds for live tracking
    
    def _get_location_from_windows_api(self) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """Try to get location using Windows Location API (requires GPS hardware)"""
        try:
            # Try Windows Location API if available
            import sys
            if sys.platform == 'win32':
                try:
                    import winrt.windows.devices.geolocation as geolocation
                    import asyncio
                    
                    async def get_location():
                        locator = geolocation.Geolocator()
                        location = await locator.get_geoposition_async()
                        return location.coordinate.latitude, location.coordinate.longitude
                    
                    # Run async function
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    lat, lon = loop.run_until_complete(get_location())
                    loop.close()
                    
                    if lat and lon:
                        # Get address using reverse geocoding
                        address = self._get_location_from_geopy(lat, lon)
                        return lat, lon, address or "GPS Location"
                except ImportError:
                    # winrt not available, skip
                    pass
                except Exception as e:
                    # GPS not available or permission denied
                    pass
        except:
            pass
        return None, None, None
    
    def _get_location_from_google_maps_api(self) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """Try to get location using more accurate IP-based services"""
        try:
            import requests
            
            # Method 1: Try ipapi.co which can provide more accurate location
            try:
                response = requests.get('https://ipapi.co/json/', timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if 'latitude' in data and 'longitude' in data:
                        lat = float(data['latitude'])
                        lon = float(data['longitude'])
                        city = data.get('city', 'Unknown')
                        region = data.get('region', 'Unknown')
                        country = data.get('country_name', 'Unknown')
                        address = f"{city}, {region}, {country}"
                        return lat, lon, address
            except:
                pass
            
            # Method 2: Try ip-api.com (free, no API key needed)
            try:
                response = requests.get('http://ip-api.com/json/', timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'success':
                        lat = float(data['lat'])
                        lon = float(data['lon'])
                        city = data.get('city', 'Unknown')
                        region = data.get('regionName', 'Unknown')
                        country = data.get('country', 'Unknown')
                        address = f"{city}, {region}, {country}"
                        return lat, lon, address
            except:
                pass
            
            return None, None, None
        except Exception as e:
            return None, None, None
    
    def _get_location_from_geopy(self, lat: float, lon: float) -> Optional[str]:
        """Get address from coordinates using geopy reverse geocoding"""
        try:
            from geopy.geocoders import Nominatim
            geolocator = Nominatim(user_agent="human_detection_system")
            location = geolocator.reverse(f"{lat}, {lon}", timeout=10)
            if location:
                return location.address
        except:
            pass
        return None
    
    def get_location(self, force_update: bool = False) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """Get current LIVE GPS location from device - Tries multiple accurate methods"""
        try:
            import geocoder
            import requests
        except ImportError:
            print_warning("Required libraries not installed. Install with: pip install geocoder requests geopy")
            return None, None, None
        
        current_time = time.time()
        
        # Always update location if forced, or if enough time has passed, or if location is None
        should_update = (force_update or 
                        self.latitude is None or 
                        self.longitude is None or 
                        self.last_update is None or 
                        (current_time - self.last_update) > self.update_interval)
        
        if should_update:
            try:
                # Method 1: Try Windows Location API for actual GPS (if available)
                lat, lon, address = self._get_location_from_windows_api()
                if lat and lon:
                    self.latitude = lat
                    self.longitude = lon
                    self.location_name = address
                    
                    # Parse address for details
                    if address and address != "GPS Location":
                        parts = address.split(', ')
                        if len(parts) >= 3:
                            self.city = parts[0] if parts[0] != 'Unknown' else None
                            self.region = parts[1] if len(parts) > 1 and parts[1] != 'Unknown' else None
                            self.country = parts[-1] if parts[-1] != 'Unknown' else None
                    
                    self.last_update = current_time
                    print_success(f"📍 GPS location from device: {lat:.6f}, {lon:.6f}")
                    return self.latitude, self.longitude, self.location_name
                
                # Method 2: Try ipapi.co/ip-api for more accurate IP-based location
                lat, lon, address = self._get_location_from_google_maps_api()
                if lat and lon:
                    self.latitude = lat
                    self.longitude = lon
                    self.location_name = address
                    
                    # Get detailed location info using geopy
                    if address:
                        parts = address.split(', ')
                        if len(parts) >= 3:
                            self.city = parts[0] if parts[0] != 'Unknown' else None
                            self.region = parts[1] if len(parts) > 1 and parts[1] != 'Unknown' else None
                            self.country = parts[-1] if parts[-1] != 'Unknown' else None
                    
                    # Try to get more accurate address using reverse geocoding
                    try:
                        detailed_address = self._get_location_from_geopy(lat, lon)
                        if detailed_address:
                            self.location_name = detailed_address
                            # Parse address components
                            addr_parts = detailed_address.split(', ')
                            if len(addr_parts) > 0:
                                self.city = addr_parts[0] if not self.city else self.city
                            if len(addr_parts) > 1:
                                self.region = addr_parts[-2] if not self.region else self.region
                            if len(addr_parts) > 0:
                                self.country = addr_parts[-1] if not self.country else self.country
                    except:
                        pass
                    
                    self.last_update = current_time
                    print_info(f"📍 Live location updated: {lat:.6f}, {lon:.6f}")
                    return self.latitude, self.longitude, self.location_name
                
                # Method 2: Try geocoder with multiple providers for better accuracy
                g = None
                providers = ['ipinfo', 'ipapi', 'freegeoip']
                
                for provider in providers:
                    try:
                        g = geocoder.ip('me', provider=provider)
                        if g.ok and g.latlng:
                            break
                    except:
                        continue
                
                # Fallback to default geocoder
                if not g or not g.ok:
                    try:
                        g = geocoder.ip('me')
                    except:
                        pass
                
                if g and g.ok and g.latlng:
                    new_lat = g.latlng[0]
                    new_lon = g.latlng[1]
                    
                    # Update location
                    self.latitude = new_lat
                    self.longitude = new_lon
                    self.location_name = g.address if hasattr(g, 'address') else None
                    
                    # Extract additional location details
                    try:
                        self.city = g.city if hasattr(g, 'city') and g.city else None
                        self.region = g.state if hasattr(g, 'state') and g.state else None
                        self.country = g.country if hasattr(g, 'country') and g.country else None
                        
                        # Try reverse geocoding for more accurate address
                        if new_lat and new_lon:
                            detailed_address = self._get_location_from_geopy(new_lat, new_lon)
                            if detailed_address:
                                self.location_name = detailed_address
                    except Exception as e:
                        pass
                    
                    self.last_update = current_time
                    print_info(f"📍 Location updated: {new_lat:.6f}, {new_lon:.6f}")
                    return self.latitude, self.longitude, self.location_name
                else:
                    print_warning("Could not determine location from any service")
                    return None, None, None
                    
            except Exception as e:
                print_warning(f"Location service error: {e}")
                return None, None, None
        
        return self.latitude, self.longitude, self.location_name
    
    def get_full_location_details(self, force_update: bool = False) -> Dict[str, Any]:
        """Get complete location details including place, lat, lon, region"""
        self.get_location(force_update=force_update)
        
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "place": self.location_name or "Unknown",
            "city": self.city or "Unknown",
            "region": self.region or "Unknown",
            "country": self.country or "Unknown"
        }
    
    def get_location_string(self) -> str:
        """Get formatted location string"""
        lat, lon, name = self.get_location()
        if lat and lon:
            return f"Lat: {lat:.6f}, Lon: {lon:.6f}"
        return "Location: Not available"
    
    def get_location_for_email(self, force_update: bool = False) -> str:
        """Get location string formatted for email with full details"""
        details = self.get_full_location_details(force_update=force_update)
        
        if details["latitude"] and details["longitude"]:
            location_text = f"📍 CAPTURE LOCATION DETAILS:\n"
            location_text += f"{'='*50}\n"
            location_text += f"Place: {details['place']}\n"
            location_text += f"City: {details['city']}\n"
            location_text += f"Region/State: {details['region']}\n"
            location_text += f"Country: {details['country']}\n"
            location_text += f"{'='*50}\n"
            location_text += f"Latitude: {details['latitude']:.6f}\n"
            location_text += f"Longitude: {details['longitude']:.6f}\n"
            location_text += f"{'='*50}\n"
            location_text += f"Google Maps: https://www.google.com/maps?q={details['latitude']},{details['longitude']}\n"
            return location_text
        return "Location: Not available"
    
    def get_location_for_image_overlay(self, force_update: bool = False) -> Dict[str, str]:
        """Get location details formatted for image overlay"""
        details = self.get_full_location_details(force_update=force_update)
        
        if details["latitude"] and details["longitude"]:
            return {
                "place": details['place'],
                "coordinates": f"Lat: {details['latitude']:.6f}, Lon: {details['longitude']:.6f}",
                "region": f"{details['city']}, {details['region']}, {details['country']}"
            }
        return {
            "place": "Location unavailable",
            "coordinates": "N/A",
            "region": "N/A"
        }


