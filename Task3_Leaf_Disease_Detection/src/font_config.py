"""
Font configuration module to handle matplotlib font issues
"""

import matplotlib
import matplotlib.pyplot as plt
import warnings

def configure_matplotlib_fonts():
    """Configure matplotlib to use safe fonts and avoid Unicode issues"""
    
    # Suppress font warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
    # Set font configuration
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = [
        'Arial', 
        'DejaVu Sans', 
        'Liberation Sans', 
        'Bitstream Vera Sans', 
        'sans-serif'
    ]
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # Set other safe defaults
    matplotlib.rcParams['figure.max_open_warning'] = 0
    matplotlib.rcParams['axes.formatter.use_mathtext'] = True
    
    # Use Agg backend if display is not available
    try:
        plt.figure()
        plt.close()
    except:
        matplotlib.use('Agg')
    
    print("[INFO] Matplotlib font configuration applied successfully")

def get_safe_symbols():
    """Return ASCII alternatives for common Unicode symbols"""
    return {
        '✅': '[OK]',
        '❌': '[ERROR]',
        '⚠️': '[WARNING]',
        '📊': '[CHART]',
        '📈': '[GRAPH]',
        '📉': '[TREND]',
        '🎨': '[VIZ]',
        '🔍': '[SEARCH]',
        '🚀': '[START]',
        '⏱️': '[TIME]',
        '📁': '[FOLDER]',
        '📋': '[LIST]',
        '🎯': '[TARGET]',
        '🔧': '[TOOL]',
        '💡': '[IDEA]',
        '🎉': '[SUCCESS]',
        '⭐': '[STAR]',
        '✓': '[OK]',
        '✗': '[X]',
        '•': '-',
        '→': '->',
        '←': '<-',
        '↑': '^',
        '↓': 'v'
    }

def replace_unicode_symbols(text):
    """Replace Unicode symbols with ASCII alternatives"""
    safe_symbols = get_safe_symbols()
    
    for unicode_char, ascii_replacement in safe_symbols.items():
        text = text.replace(unicode_char, ascii_replacement)
    
    return text

# Apply configuration when module is imported
configure_matplotlib_fonts()