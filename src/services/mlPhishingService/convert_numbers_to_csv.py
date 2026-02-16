"""
Convert Apple Numbers (.numbers) file to CSV format.
This script reads a .numbers file and converts it to CSV for training.
"""

import os
import sys
import pandas as pd
from pathlib import Path

try:
    from numbers_parser import Document
    NUMBERS_PARSER_AVAILABLE = True
except ImportError:
    NUMBERS_PARSER_AVAILABLE = False
    print("Warning: numbers-parser not installed. Trying alternative methods...")


def convert_numbers_to_csv_using_parser(numbers_path: str, csv_path: str) -> bool:
    """Convert .numbers to CSV using numbers-parser library."""
    try:
        doc = Document(numbers_path)
        sheets = doc.sheets
        tables = sheets[0].tables
        table = tables[0]
        
        # Get all rows
        rows = []
        for row in table.rows():
            row_data = []
            for cell in row:
                value = cell.value
                # Handle None values
                if value is None:
                    value = ""
                row_data.append(value)
            rows.append(row_data)
        
        if not rows:
            print("❌ No data found in .numbers file")
            return False
        
        # First row is headers
        headers = rows[0]
        data = rows[1:]
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=headers)
        
        # Save to CSV
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"✅ Successfully converted using numbers-parser")
        print(f"   Converted {len(df)} rows with {len(df.columns)} columns")
        return True
    except Exception as e:
        print(f"❌ Error using numbers-parser: {e}")
        import traceback
        traceback.print_exc()
        return False


def convert_numbers_to_csv_alternative(numbers_path: str, csv_path: str) -> bool:
    """Alternative method: Try using pandas with different engines."""
    try:
        # Try reading as Excel (sometimes works if file is actually xlsx)
        try:
            df = pd.read_excel(numbers_path, engine='openpyxl')
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"✅ Successfully converted using openpyxl (Excel format)")
            return True
        except:
            pass
        
        # Try reading as Excel with xlrd engine
        try:
            df = pd.read_excel(numbers_path, engine='xlrd')
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"✅ Successfully converted using xlrd")
            return True
        except:
            pass
        
        print("❌ Could not read .numbers file with pandas")
        return False
    except Exception as e:
        print(f"❌ Error in alternative method: {e}")
        return False


def convert_numbers_to_csv(numbers_path: str, csv_path: str = None) -> str:
    """
    Convert .numbers file to CSV.
    
    Args:
        numbers_path: Path to .numbers file
        csv_path: Optional output CSV path (default: same name with .csv extension)
        
    Returns:
        Path to created CSV file
    """
    numbers_path = Path(numbers_path)
    
    if not numbers_path.exists():
        raise FileNotFoundError(f"Numbers file not found: {numbers_path}")
    
    if csv_path is None:
        csv_path = numbers_path.with_suffix('.csv')
    else:
        csv_path = Path(csv_path)
    
    print(f"Converting {numbers_path} to {csv_path}...")
    
    # Try numbers-parser first (best method)
    if NUMBERS_PARSER_AVAILABLE:
        if convert_numbers_to_csv_using_parser(str(numbers_path), str(csv_path)):
            return str(csv_path)
    
    # Try alternative methods
    if convert_numbers_to_csv_alternative(str(numbers_path), str(csv_path)):
        return str(csv_path)
    
    # If all methods fail, provide instructions
    print("\n" + "=" * 70)
    print("❌ Could not convert .numbers file automatically")
    print("=" * 70)
    print("\nPlease try one of these options:")
    print("\n1. Install numbers-parser library:")
    print("   pip install numbers-parser")
    print("\n2. Manually export from Numbers app:")
    print("   - Open the file in Numbers")
    print("   - File > Export To > CSV")
    print("\n3. Use online converter or command-line tool")
    print("=" * 70)
    
    raise RuntimeError("Could not convert .numbers file to CSV")


def main():
    """Main function for command-line usage."""
    if len(sys.argv) < 2:
        print("Usage: python convert_numbers_to_csv.py <input.numbers> [output.csv]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        csv_path = convert_numbers_to_csv(input_path, output_path)
        print(f"\n✅ Conversion complete!")
        print(f"   CSV saved to: {csv_path}")
        print(f"   Rows: {len(pd.read_csv(csv_path))}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
