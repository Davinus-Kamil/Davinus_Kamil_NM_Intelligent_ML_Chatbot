import os

def get_base_path():
    """Get the absolute path to the project directory."""
    return os.path.dirname(os.path.abspath(__file__))

def get_model_path():
    """Get the absolute path to the model file."""
    base_path = get_base_path()
    model_path = os.path.join(base_path, 'chatbot_model.joblib')
    # Normalize the path to handle any potential issues with separators
    return os.path.normpath(model_path)

def get_vectorizer_path():
    """Get the absolute path to the vectorizer file."""
    base_path = get_base_path()
    vectorizer_path = os.path.join(base_path, 'vectorizer.joblib')
    # Normalize the path to handle any potential issues with separators
    return os.path.normpath(vectorizer_path)

def get_data_path():
    """Get the absolute path to the data file."""
    base_path = get_base_path()
    data_path = os.path.join(base_path, 'customer_queries.csv')
    # Normalize the path to handle any potential issues with separators
    return os.path.normpath(data_path)

def check_files_exist():
    """Check if all required files exist and get their details."""
    files = {
        'Model': get_model_path(),
        'Vectorizer': get_vectorizer_path(),
        'Data': get_data_path()
    }
    
    status = {}
    for name, path in files.items():
        exists = os.path.exists(path)
        try:
            size = os.path.getsize(path) if exists else 0
        except Exception:
            size = 0
            
        status[name] = {
            'exists': exists,
            'path': path,
            'size': size,
            'is_absolute': os.path.isabs(path),
            'normalized_path': os.path.normpath(path)
        }
    
    return status

def print_debug_info():
    """Print debug information about paths and files."""
    print("\n=== Debug Information ===")
    print(f"Base Path: {get_base_path()}")
    print(f"Current Working Directory: {os.getcwd()}")
    
    status = check_files_exist()
    for name, info in status.items():
        print(f"\n{name}:")
        print(f"  Path: {info['path']}")
        print(f"  Exists: {info['exists']}")
        print(f"  Size: {info['size']} bytes")
        print(f"  Is Absolute: {info['is_absolute']}")
        print(f"  Normalized: {info['normalized_path']}")

if __name__ == "__main__":
    print_debug_info() 