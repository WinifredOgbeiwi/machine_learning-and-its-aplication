import sys
import importlib.metadata
print(f"Python version:\n\n {sys.version}")

print("\nInstalled packages:\n")
for package in importlib.metadata.distributions():
    print(f" {package.metadata['Name']} {package.metadata['Version']}")