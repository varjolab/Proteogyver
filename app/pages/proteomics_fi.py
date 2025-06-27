"""Embedded page for Proteomics.fi."""

import dash
from dash import html

dash.register_page(__name__, path='/proteomics-fi')

layout = html.Div([
    html.Embed(
        src="http://www.proteomics.fi",
        style={
            'position': 'fixed',
            'top': '85px',  # Matches navbar height
            'left': '0',
            'bottom': '0',
            'right': '0',
            'width': '100%',
            'height': 'calc(100% - 85px)',  # Subtract navbar height
            'border': 'none',
            'margin': '0',
            'padding': '0',
            'overflow': 'hidden',
            'z-index': '999999'
        }
    )
])
