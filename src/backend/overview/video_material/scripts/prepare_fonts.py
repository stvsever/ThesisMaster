#!/usr/bin/env python3
"""Create static Manrope instances for matching browser and native typography.
Optional maintainer tool: requires fontTools, not needed to run or render the studio.
"""
from pathlib import Path
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
root = Path(__file__).resolve().parents[1] / 'assets/fonts'
for weight, style in [(400,'Regular'),(500,'Medium'),(600,'SemiBold'),(700,'Bold')]:
    font = instantiateVariableFont(TTFont(root / 'Manrope.ttf'), {'wght': weight}, inplace=False)
    font['OS/2'].usWeightClass = weight
    font['OS/2'].fsSelection = 32 if weight == 700 else 64
    font['head'].macStyle = 1 if weight == 700 else 0
    for name in font['name'].names:
        if name.nameID in (1, 16): name.string = 'Manrope'.encode(name.getEncoding())
        if name.nameID in (2, 17): name.string = style.encode(name.getEncoding())
        if name.nameID == 4: name.string = f'Manrope {style}'.encode(name.getEncoding())
        if name.nameID == 6: name.string = f'Manrope-{style}'.encode(name.getEncoding())
    font.save(root / f'Manrope-{style}.ttf')
