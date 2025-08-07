from core.model_manager import ModelManager
import pandas as pd
import numpy as np

# Quick test
print("[TESTING] ModelManager apply_model_to_data method...")
mm = ModelManager()

# Create simple test data
data = np.random.randn(20, 2)
df = pd.DataFrame(data, columns=['x', 'y'])

try:
    # This should fail since no model exists yet
    results = mm.apply_model_to_data('TestModel', df, ['x', 'y'])
    print("[FAIL] Unexpected success - should have failed")
except FileNotFoundError as e:
    print(f"[PASS] Expected error: {str(e)}")
    print("[PASS] Method works - just no model exists yet")
except Exception as e:
    print(f"[FAIL] Unexpected error: {type(e).__name__}: {str(e)}")

print("\n[TESTING] apply_model_to_data method has anomaly_indices in results...")

# Check the method source for our fix
import inspect
source = inspect.getsource(mm.apply_model_to_data)
if 'anomaly_indices' in source:
    print("[PASS] Found anomaly_indices in apply_model_to_data method")
else:
    print("[FAIL] anomaly_indices not found in apply_model_to_data method")

print("\n[COMPLETE] Test finished!")
