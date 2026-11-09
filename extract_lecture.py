"""Preserve slide text and linked speaker notes for traceability."""
import argparse
from pathlib import Path
import posixpath
import xml.etree.ElementTree as E
from zipfile import ZipFile

def main():
    p = argparse.ArgumentParser()
    p.add_argument('presentation', type=Path)
    args = p.parse_args()
    ns = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
    out = []
    with ZipFile(args.presentation) as z:
        count = len([n for n in z.namelist() if n.startswith('ppt/slides/slide') and n.endswith('.xml')])
        for i in range(1, count + 1):
            out += [f'## Слайд {i}', '\n'.join(t.text or '' for t in E.fromstring(z.read(f'ppt/slides/slide{i}.xml')).findall('.//a:t', ns))]
            for rel in E.fromstring(z.read(f'ppt/slides/_rels/slide{i}.xml.rels')):
                if rel.attrib['Type'].endswith('/notesSlide'):
                    target = posixpath.normpath(posixpath.join('ppt/slides', rel.attrib['Target']))
                    out += ['Заметки:', '\n'.join(t.text or '' for t in E.fromstring(z.read(target)).findall('.//a:t', ns))]
    target = Path(__file__).parent / 'sources' / 'lecture_notes.md'
    target.parent.mkdir(exist_ok=True)
    target.write_text('\n\n'.join(out), encoding='utf-8')
    print(f'Extracted {count} slides with linked notes')

if __name__ == '__main__':
    main()
