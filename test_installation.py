#!/usr/bin/env python3
"""
Installation Verification Script
Tests that all required packages are installed correctly for Python 3.11.14
"""

import sys

def test_imports():
    """Test importing all required packages"""
    
    packages = [
        # Core
        ('streamlit', 'Streamlit', True),
        ('pandas', 'Pandas', True),
        ('numpy', 'NumPy', True),
        
        # Visualization
        ('plotly', 'Plotly', True),
        ('folium', 'Folium', True),
        ('matplotlib', 'Matplotlib', True),
        ('seaborn', 'Seaborn', True),
        
        # Geospatial
        ('geopandas', 'GeoPandas', True),
        ('shapely', 'Shapely', True),
        ('geopy', 'Geopy', True),
        
        # Machine Learning
        ('tensorflow', 'TensorFlow', True),
        ('sklearn', 'scikit-learn', True),
        ('mlflow', 'MLflow', True),
        
        # APIs
        ('openaq', 'OpenAQ', True),
        ('requests', 'Requests', True),
        ('pyowm', 'PyOWM', True),
        
        # Utilities
        ('dotenv', 'python-dotenv', True),
        ('joblib', 'joblib', True),
        
        # Optional - Notifications
        ('sendgrid', 'SendGrid', False),
        ('twilio', 'Twilio', False),
        
        # Optional - Database
        ('sqlalchemy', 'SQLAlchemy', False),
        ('redis', 'Redis', False),
    ]
    
    print("=" * 70)
    print("PACKAGE INSTALLATION VERIFICATION")
    print("=" * 70)
    print(f"Python Version: {sys.version}")
    print("=" * 70)
    
    required_failed = []
    optional_failed = []
    
    for module, name, required in packages:
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'installed')
            status = "✅"
            print(f"{status} {name:25} {version:15} {'(required)' if required else '(optional)'}")
        except ImportError as e:
            status = "❌"
            print(f"{status} {name:25} {'NOT INSTALLED':15} {'(required)' if required else '(optional)'}")
            if required:
                required_failed.append((name, str(e)))
            else:
                optional_failed.append((name, str(e)))
    
    print("=" * 70)
    
    # Summary
    total_required = sum(1 for _, _, req in packages if req)
    total_optional = sum(1 for _, _, req in packages if not req)
    
    print(f"\nSUMMARY:")
    print(f"  Required packages: {total_required - len(required_failed)}/{total_required} installed")
    print(f"  Optional packages: {total_optional - len(optional_failed)}/{total_optional} installed")
    
    if required_failed:
        print(f"\n❌ FAILED REQUIRED PACKAGES ({len(required_failed)}):")
        for name, error in required_failed:
            print(f"  - {name}")
            print(f"    Error: {error}")
        print(f"\nTo install missing packages:")
        print(f"  pip install -r requirements.txt")
        return False
    
    if optional_failed:
        print(f"\n⚠️  OPTIONAL PACKAGES NOT INSTALLED ({len(optional_failed)}):")
        for name, error in optional_failed:
            print(f"  - {name}")
        print(f"\nThese are optional. Install if needed:")
        print(f"  pip install sendgrid twilio firebase-admin sqlalchemy redis")
    
    print(f"\n✅ All required packages installed successfully!")
    return True


def test_python_version():
    """Verify Python version"""
    version = sys.version_info
    print(f"\nPython Version Check:")
    print(f"  Current: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor == 11:
        print(f"  ✅ Python 3.11.x detected - Compatible!")
        return True
    else:
        print(f"  ⚠️  Warning: This project is optimized for Python 3.11.x")
        print(f"     Your version may work but hasn't been fully tested.")
        return True


def test_tensorflow():
    """Test TensorFlow installation and GPU support"""
    try:
        import tensorflow as tf
        print(f"\nTensorFlow Check:")
        print(f"  Version: {tf.__version__}")
        
        # Check GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"  ✅ GPU support detected: {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                print(f"     GPU {i}: {gpu.name}")
        else:
            print(f"  ℹ️  No GPU detected - will use CPU (slower but works)")
        
        # Quick test
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        print(f"  ✅ TensorFlow working correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ TensorFlow test failed: {str(e)}")
        return False


def test_streamlit():
    """Test Streamlit"""
    try:
        import streamlit as st
        print(f"\nStreamlit Check:")
        print(f"  Version: {st.__version__}")
        print(f"  ✅ Streamlit ready")
        print(f"  Run dashboard with: streamlit run app.py")
        return True
    except Exception as e:
        print(f"\n❌ Streamlit test failed: {str(e)}")
        return False


def test_geopandas():
    """Test GeoPandas"""
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        
        print(f"\nGeoPandas Check:")
        print(f"  Version: {gpd.__version__}")
        
        # Quick test
        gdf = gpd.GeoDataFrame(
            {'name': ['Delhi']},
            geometry=[Point(77.2090, 28.6139)],
            crs='EPSG:4326'
        )
        print(f"  ✅ GeoPandas working correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ GeoPandas test failed: {str(e)}")
        print(f"  Install GDAL first:")
        print(f"    Ubuntu: sudo apt-get install gdal-bin libgdal-dev")
        print(f"    macOS: brew install gdal")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("DELHI AQI DASHBOARD - INSTALLATION VERIFICATION")
    print("=" * 70 + "\n")
    
    results = []
    
    # Test Python version
    results.append(("Python Version", test_python_version()))
    
    # Test imports
    results.append(("Package Imports", test_imports()))
    
    # Test specific packages
    results.append(("TensorFlow", test_tensorflow()))
    results.append(("Streamlit", test_streamlit()))
    results.append(("GeoPandas", test_geopandas()))
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    all_passed = all(result for _, result in results)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status:12} - {test_name}")
    
    print("=" * 70)
    
    if all_passed:
        print("\n🎉 All tests passed! Your environment is ready.")
        print("\nNext steps:")
        print("  1. Configure API keys in .streamlit/secrets.toml")
        print("  2. Run: python data_pipeline.py")
        print("  3. Run: streamlit run app.py")
        return 0
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("  - Update pip: pip install --upgrade pip")
        print("  - Reinstall: pip install -r requirements.txt")
        print("  - Check GDAL: For GeoPandas issues")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
