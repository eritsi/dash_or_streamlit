import streamlit as st
import base64
import dtreeviz

def svg_write(svg, center=True):
    """
    Disable center to left-margin align like other objects.
    """
    with open(svg, "rb") as f:
        svg_data = f.read()
    
    # Encode as base 64
    b64 = base64.b64encode(svg_data).decode("utf-8")

    # Add some CSS on top
    css_justify = "center" if center else "left"
    css = f'<p style="text-align:center; display: flex; justify-content: {css_justify};">'
    html = f'{css}<img src="data:image/svg+xml;base64,{b64}"/>'

    # Write the HTML
    st.write(html, unsafe_allow_html=True)

import svgwrite
from xml.dom import minidom

def get_svg_size(file_path):
    svg = minidom.parse(file_path)
    width = svg.documentElement.getAttribute("width").replace('pt','')
    height = svg.documentElement.getAttribute("height").replace('pt','')
    return float(width), float(height)

def combine_svgs(file1, file2, output_file):
    width1, height1 = get_svg_size(file1)
    width2, height2 = get_svg_size(file2)
    
    total_width = max(width1, width2)
    total_height = height1 + height2
    
    dwg = svgwrite.Drawing(output_file, size=(total_width, total_height))
    
    dwg.add(dwg.image(href=file1, insert=(0, 0), size=(width1, height1)))
    dwg.add(dwg.image(href=file2, insert=(0, height1), size=(width2, height2)))
    
    dwg.save()

if inp := st.chat_input("path"):
    inp2 = inp.split('.')[0]+"_score.svg"
    output_file = inp

    # combine_svgs(inp, inp2, output_file)
    svg_write(inp)