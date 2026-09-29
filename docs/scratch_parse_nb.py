import json
import codecs

def parse_notebook():
    with codecs.open('final_model/notebook/__notebook_source__.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    with codecs.open('final_model/notebook/extracted_source.py', 'w', encoding='utf-8') as out:
        for cell in nb.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                out.write(source + '\n\n')

if __name__ == '__main__':
    parse_notebook()
