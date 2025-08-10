import json
import sys
from typing import Any, Dict, List, Union
from collections import defaultdict
from pathlib import Path

class JSONStructureAnalyzer:
    def __init__(self):
        self.type_counts = defaultdict(int)
        self.max_depth = 0
        self.sample_values = {}
        
    def analyze_structure(self, data: Any, path: str = "root", depth: int = 0) -> Dict[str, Any]:
        """Analyze JSON structure and return LLM-friendly representation"""
        self.max_depth = max(self.max_depth, depth)
        
        if isinstance(data, dict):
            return self._analyze_object(data, path, depth)
        elif isinstance(data, list):
            return self._analyze_array(data, path, depth)
        else:
            return self._analyze_primitive(data, path)
    
    def _analyze_object(self, obj: Dict, path: str, depth: int) -> Dict[str, Any]:
        """Analyze dictionary/object structure"""
        self.type_counts['object'] += 1
        
        structure = {
            'type': 'object',
            'path': path,
            'properties': {},
            'required_fields': list(obj.keys()),
            'optional_fields': [],
            'property_count': len(obj)
        }
        
        for key, value in obj.items():
            property_path = f"{path}.{key}"
            structure['properties'][key] = self.analyze_structure(value, property_path, depth + 1)
        
        return structure
    
    def _analyze_array(self, arr: List, path: str, depth: int) -> Dict[str, Any]:
        """Analyze array structure"""
        self.type_counts['array'] += 1
        
        structure = {
            'type': 'array',
            'path': path,
            'length': len(arr),
            'item_types': set(),
            'items_structure': None
        }
        
        if not arr:
            structure['items_structure'] = {'type': 'empty_array'}
            return structure
        
        # Analyze array items
        item_structures = []
        for i, item in enumerate(arr[:5]):  # Analyze first 5 items for pattern
            item_path = f"{path}[{i}]"
            item_structure = self.analyze_structure(item, item_path, depth + 1)
            item_structures.append(item_structure)
            structure['item_types'].add(item_structure['type'])
        
        structure['item_types'] = list(structure['item_types'])
        
        # Determine if array has consistent structure
        if len(structure['item_types']) == 1:
            structure['items_structure'] = item_structures[0]
            structure['homogeneous'] = True
        else:
            structure['items_structure'] = item_structures
            structure['homogeneous'] = False
        
        return structure
    
    def _analyze_primitive(self, value: Any, path: str) -> Dict[str, Any]:
        """Analyze primitive values"""
        value_type = type(value).__name__
        self.type_counts[value_type] += 1
        
        structure = {
            'type': value_type,
            'path': path,
            'value': value if len(str(value)) < 100 else f"{str(value)[:97]}...",
            'length': len(str(value)) if isinstance(value, str) else None
        }
        
        # Add type-specific information
        if isinstance(value, str):
            structure['string_length'] = len(value)
            structure['is_empty'] = value == ""
        elif isinstance(value, (int, float)):
            structure['numeric_value'] = value
        elif isinstance(value, bool):
            structure['boolean_value'] = value
        elif value is None:
            structure['is_null'] = True
        
        return structure
    
    def generate_llm_friendly_output(self, structure: Dict[str, Any]) -> str:
        """Generate LLM-friendly text representation"""
        output = []
        output.append("JSON STRUCTURE ANALYSIS")
        output.append("=" * 50)
        
        # Summary
        output.append(f"Maximum nesting depth: {self.max_depth}")
        output.append(f"Type distribution: {dict(self.type_counts)}")
        output.append("")
        
        # Detailed structure
        output.append("DETAILED STRUCTURE:")
        output.append("-" * 30)
        self._format_structure(structure, output, indent=0)
        
        # Programming hints
        output.append("\nPROGRAMMING HINTS:")
        output.append("-" * 30)
        self._generate_programming_hints(structure, output)
        
        return "\n".join(output)
    
    def _format_structure(self, structure: Dict[str, Any], output: List[str], indent: int = 0):
        """Format structure for LLM readability"""
        spaces = "  " * indent
        
        if structure['type'] == 'object':
            output.append(f"{spaces}📁 OBJECT at {structure['path']}")
            output.append(f"{spaces}   Properties: {structure['property_count']}")
            output.append(f"{spaces}   Required fields: {structure['required_fields']}")
            
            for prop_name, prop_structure in structure['properties'].items():
                output.append(f"{spaces}   └── {prop_name}:")
                self._format_structure(prop_structure, output, indent + 2)
        
        elif structure['type'] == 'array':
            output.append(f"{spaces}📋 ARRAY at {structure['path']}")
            output.append(f"{spaces}   Length: {structure['length']}")
            output.append(f"{spaces}   Item types: {structure['item_types']}")
            output.append(f"{spaces}   Homogeneous: {structure.get('homogeneous', False)}")
            
            if structure['items_structure'] and structure['items_structure']['type'] != 'empty_array':
                output.append(f"{spaces}   Items structure:")
                if isinstance(structure['items_structure'], list):
                    for i, item_struct in enumerate(structure['items_structure']):
                        output.append(f"{spaces}     Item {i}:")
                        self._format_structure(item_struct, output, indent + 3)
                else:
                    self._format_structure(structure['items_structure'], output, indent + 2)
        
        else:
            # Primitive type
            output.append(f"{spaces}🔹 {structure['type'].upper()} at {structure['path']}")
            if 'value' in structure:
                output.append(f"{spaces}   Value: {structure['value']}")
            if 'string_length' in structure:
                output.append(f"{spaces}   Length: {structure['string_length']}")
    
    def _generate_programming_hints(self, structure: Dict[str, Any], output: List[str]):
        """Generate programming-specific hints"""
        output.append("• Access patterns:")
        self._generate_access_patterns(structure, output, "data")
        
        output.append("\n• Validation checks needed:")
        self._generate_validation_hints(structure, output)
        
        output.append("\n• Potential issues:")
        self._generate_potential_issues(structure, output)
    
    def _generate_access_patterns(self, structure: Dict[str, Any], output: List[str], var_name: str):
        """Generate code access patterns"""
        if structure['type'] == 'object':
            for prop_name, prop_structure in structure['properties'].items():
                access_pattern = f"{var_name}['{prop_name}']"
                output.append(f"  - {access_pattern} → {prop_structure['type']}")
                
                if prop_structure['type'] in ['object', 'array']:
                    self._generate_access_patterns(prop_structure, output, access_pattern)
        
        elif structure['type'] == 'array':
            if structure['items_structure'] and structure['items_structure']['type'] != 'empty_array':
                access_pattern = f"{var_name}[i]"
                item_type = structure['items_structure']['type']
                output.append(f"  - {access_pattern} → {item_type}")
    
    def _generate_validation_hints(self, structure: Dict[str, Any], output: List[str]):
        """Generate validation hints"""
        if structure['type'] == 'array':
            output.append(f"  - Check if array is empty before accessing items")
            if not structure.get('homogeneous', True):
                output.append(f"  - Array contains mixed types: {structure['item_types']}")
        
        elif structure['type'] == 'object':
            output.append(f"  - Verify required fields exist: {structure['required_fields']}")
            
            for prop_name, prop_structure in structure['properties'].items():
                if prop_structure['type'] == 'NoneType':
                    output.append(f"  - Field '{prop_name}' can be null")
    
    def _generate_potential_issues(self, structure: Dict[str, Any], output: List[str]):
        """Generate potential programming issues"""
        if self.max_depth > 5:
            output.append("  - Deep nesting detected - consider flattening")
        
        if self.type_counts.get('NoneType', 0) > 0:
            output.append("  - Null values present - add null checks")
        
        if any(count > 100 for count in self.type_counts.values()):
            output.append("  - Large data structure - consider pagination/chunking")

def analyze_json_file(file_path: str) -> str:
    """Main function to analyze JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        analyzer = JSONStructureAnalyzer()
        structure = analyzer.analyze_structure(data)
        return analyzer.generate_llm_friendly_output(structure)
    
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found"
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format - {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

# Example usage
if __name__ == "__main__":

    
    file_path = 'data/new_processed_gait_data#39_1.json'
    if not Path(file_path).exists():
        print(f"File {file_path} does not exist. Please check the path.")
        sys.exit(1)

    result = analyze_json_file(file_path)
    print(result)
    
    # Optional: Save to file
    output_file = f"{file_path}_structure_1.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result)
    print(f"\nStructure analysis saved to: {output_file}")
