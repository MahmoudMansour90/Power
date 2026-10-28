"""
Shared choropleth-map styling for every pillar's page, so the maps look and
behave consistently app-wide (same state-border treatment, same "no data"
convention, same contiguous-US-only framing, same year-selection behavior).

v3.1 REWRITE -- this file replaces the earlier animated-frame map entirely,
in direct response to real bugs the user found after deploying v3:

  1. "State boundaries on top of counties, masking everything, no clicks
     possible on non-default years" -- root cause: the old
     build_animated_choropleth() only attached `geojson=` to the BASE trace;
     every animation frame for every other year had no geojson at all, so
     Plotly drew no county shapes (and nothing to click) for any year except
     the one shown on load. Fix: animation frames are gone. Every year now
     gets its own fully-formed, freshly-built static Choropleth trace,
     rebuilt on each Streamlit rerun when the year selector changes.
  2. "Alaska and Hawaii are still shown on the map" -- root cause: Plotly's
     `scope='usa'` basemap draws AK/HI inset boxes as a fixed feature of
     that basemap, regardless of lataxis_range/lonaxis_range cropping or of
     whether geojson data exists there. Fix: Plotly's built-in basemap is
     turned off entirely (`visible=False`). The map now shows ONLY our own
     geometry: the county choropleth (geojson already has AK/HI features
     dropped -- see data_prep.py) and our own dissolved state-border lines
     (also built only from the AK/HI-free county geometry -- see
     build_state_borders() in data_prep.py). With no basemap and no AK/HI
     geometry anywhere in either trace, Alaska and Hawaii are now
     structurally impossible to show, not just hidden.
  3. State borders are our OWN `go.Scattergeo` line trace (drawn from
     dissolved county polygons), not Plotly's `showsubunits` layer -- this
     is what makes them visible without pulling in scope='usa''s basemap.
  4. The year control is a plain `st.select_slider` (every single year still
     selectable, full resolution) rendered by Streamlit itself, paired with
     a small HTML "year ruler" underneath it (render_year_ruler) that marks
     every 2nd year and colors forecast-year tick labels differently from
     actual-year ones. Plotly's own slider/animation is no longer used at
     all, so this is the mechanism for both the "mark every two years" and
     "color the projected years differently" requests.
  5. "No data" is shown without relying on Plotly's `landcolor` fallback
     (which only existed because of the now-removed basemap): the
     choropleth trace covers EVERY contiguous county every year, with
     counties missing that year's data assigned a sentinel z-value that
     falls below the real data range, and a remapped colorscale that
     reserves a small solid-gray band at the bottom of its range for that
     sentinel, with the real named colorscale compressed into the rest.
  6. No chart download affordances: the modebar's camera/"download as PNG"
     button is removed on every plotly_chart call in the app (see
     NO_DOWNLOAD_CONFIG below); table CSV/clipboard export is disabled
     globally via .streamlit/config.toml (client.disableDataExport).
"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go

NO_DATA_COLOR = '#d9d9d9'   # distinct, visible gray -- not white/background
FORECAST_TICK_COLOR = '#e35b1e'   # distinct from actual-year ticks
ACTUAL_TICK_COLOR = '#444444'

# Plotly config applied to EVERY st.plotly_chart call in the app: removes
# the camera/"download plot as png" modebar button. Table CSV/clipboard
# export is a separate, global Streamlit setting -- see
# .streamlit/config.toml (client.disableDataExport = true).
NO_DOWNLOAD_CONFIG = {'modeBarButtonsToRemove': ['toImage'], 'displaylogo': False}

# Named colorscale stops (sequential.YlOrRd / sequential.OrRd from
# plotly.express.colors, confirmed directly against the installed plotly
# version rather than assumed) -- hardcoded here so _remap_colorscale()
# doesn't depend on importing plotly.express just for two small lists.
_YLORRD = ['rgb(255,255,204)', 'rgb(255,237,160)', 'rgb(254,217,118)', 'rgb(254,178,76)',
           'rgb(253,141,60)', 'rgb(252,78,42)', 'rgb(227,26,28)', 'rgb(189,0,38)', 'rgb(128,0,38)']
_ORRD = ['rgb(255,247,236)', 'rgb(254,232,200)', 'rgb(253,212,158)', 'rgb(253,187,132)',
         'rgb(252,141,89)', 'rgb(239,101,72)', 'rgb(215,48,31)', 'rgb(179,0,0)', 'rgb(127,0,0)']
_NAMED_COLORSCALES = {'YlOrRd': _YLORRD, 'OrRd': _ORRD}


def _remap_colorscale(name, gray_frac=0.15):
    """Compress a named sequential colorscale into the [gray_frac, 1] slice
    of the [0,1] colorscale domain, and reserve [0, gray_frac] as a single
    solid NO_DATA_COLOR band. Used together with a sentinel z-value (see
    _with_sentinel below) so one single Choropleth trace can show both
    real data and a clear "no data" gray -- without Plotly's landcolor
    fallback, which only existed because of the basemap we've now turned
    off entirely (visible=False) to make AK/HI structurally impossible."""
    stops = _NAMED_COLORSCALES[name]
    n = len(stops)
    scale = [[0.0, NO_DATA_COLOR], [max(gray_frac - 1e-6, 0.0), NO_DATA_COLOR]]
    for i, color in enumerate(stops):
        frac = i / (n - 1)
        pos = gray_frac + frac * (1 - gray_frac)
        scale.append([pos, color])
    scale[-1][0] = 1.0  # avoid float drift leaving the last stop short of 1.0
    return scale


def _with_sentinel(df, color_col, vmin, vmax, gray_frac=0.15):
    """Given a dataframe that already has one row per county IN SCOPE for
    the year (including counties with no reported value, i.e. NaN in
    color_col), return (z_values, zmin, zmax) where missing rows get a
    sentinel value that lands inside the gray band of a colorscale built
    by _remap_colorscale(..., gray_frac), and real values land in the
    [gray_frac, 1] band unchanged relative to each other."""
    span = (vmax - vmin) or 1.0
    zmin = vmin - (gray_frac / (1 - gray_frac)) * span
    zmax = vmax
    sentinel = zmin + 0.25 * (vmin - zmin)  # solidly inside the gray band, not at its edge
    z = df[color_col].fillna(sentinel)
    return z, zmin, zmax


def build_static_choropleth(df, geojson, state_borders, color_col, color_scale,
                             vmin, vmax, hover_text, colorbar_title, gray_frac=0.15):
    """Build ONE static (non-animated) contiguous-US county choropleth for
    a single year, re-built fresh on every Streamlit rerun when the year
    selector changes.

    df must have one row per county FIPS IN SCOPE (i.e. every contiguous
    county, not just the ones with real data that year -- see each
    pillar's _year_df() helper) with columns ['FIPS', color_col]. hover_text
    is a precomputed per-row text array/Series aligned to df's row order,
    expected to already say something sensible ("no data reported") for
    rows where color_col is NaN.

    geojson: the shared, AK/HI-free county boundary file.
    state_borders: dict with 'lons'/'lats' lists (None-separated rings),
    produced by data_prep.py's build_state_borders() from the SAME AK/HI-free
    county geometry -- so state lines, like the choropleth itself, can never
    include Alaska or Hawaii.
    """
    z, zmin, zmax = _with_sentinel(df, color_col, vmin, vmax, gray_frac)
    remapped = _remap_colorscale(color_scale, gray_frac)

    choropleth = go.Choropleth(
        geojson=geojson, locations=df['FIPS'], z=z,
        zmin=zmin, zmax=zmax, colorscale=remapped,
        hovertext=hover_text, hoverinfo='text',
        marker_line_width=0, marker_line_color='rgba(0,0,0,0)',
        colorbar=dict(
            title=colorbar_title, thickness=16, len=0.75,
            # Real data only occupies [gray_frac, 1] of the bar -- cropping
            # the displayed tick range keeps the colorbar itself from
            # showing a confusing "below vmin" gray sliver with no label.
            tickmode='auto',
        ),
    )
    # Colorbar still spans the full [zmin, zmax] internally (that's what
    # the gray sentinel band needs), but we only want tick labels drawn
    # across the real [vmin, vmax] portion -- Plotly can't crop a
    # colorbar's drawn range independent of its trace's zmin/zmax, so
    # instead we fix explicit tick values across the real range only.
    tick_vals = list(np.linspace(vmin, vmax, 5))
    choropleth.colorbar.tickvals = tick_vals
    choropleth.colorbar.ticktext = [f'{v:.0f}' if abs(v) >= 10 else f'{v:.1f}' for v in tick_vals]

    borders = go.Scattergeo(
        lon=state_borders['lons'], lat=state_borders['lats'], mode='lines',
        line=dict(width=1.8, color='rgba(25,25,25,0.9)'),
        hoverinfo='skip', showlegend=False,
    )

    fig = go.Figure(data=[choropleth, borders])
    # NOTE: visible=False turns off Plotly's drawn basemap shapes (land,
    # countries, lakes, AK/HI insets) -- it does NOT stop Plotly from
    # fetching cdn.plot.ly's base topology file for the chosen projection
    # at render time; that fetch happens for every projection_type, not
    # just 'albers usa' (confirmed directly: an identical blocked request
    # occurs with 'mercator' too, just for a different file). Since no
    # AK/HI coordinates exist anywhere in our own geojson or state-border
    # data, Albers USA's built-in insets are never invoked regardless, so
    # it's kept for its better-proportioned look of the contiguous US.
    fig.update_geos(
        visible=False, projection_type='albers usa',
        fitbounds='geojson', bgcolor='rgba(0,0,0,0)',
    )
    fig.update_layout(
        height=560, margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def render_year_slider(key, years, default_year, last_actual_year):
    """The actual, draggable year control -- a plain st.select_slider over
    EVERY year (full resolution, same as before), paired with an HTML
    'ruler' underneath marking every 2nd year and coloring forecast-year
    labels differently from actual-year ones (Plotly's own slider/animation
    is no longer used anywhere, so this replaces its tick-marking and
    per-step coloring, neither of which Plotly's animate-slider API
    actually supports natively for per-step font color)."""
    default_year = default_year if default_year in years else years[-1]
    sel_year = st.select_slider(
        'Year', options=years, value=default_year, key=key,
    )
    render_year_ruler(years, last_actual_year, sel_year)
    return sel_year


def render_year_ruler(years, last_actual_year, current_year):
    """Visual-only HTML strip: a tick + label every 2 years, forecast-year
    labels in a distinct color, with the currently-selected year bolded.
    Purely informational -- the st.select_slider above it is what's
    actually draggable, and already supports every single year."""
    ymin, ymax = years[0], years[-1]
    span = (ymax - ymin) or 1
    marks = [y for y in years if (y - ymin) % 2 == 0]
    if years[-1] not in marks:
        marks.append(years[-1])
    ticks_html = []
    for y in marks:
        pct = (y - ymin) / span * 100
        is_forecast = y > last_actual_year
        color = FORECAST_TICK_COLOR if is_forecast else ACTUAL_TICK_COLOR
        weight = '700' if y == current_year else '400'
        underline = 'text-decoration:underline;' if y == current_year else ''
        ticks_html.append(
            f'<div style="position:absolute; left:{pct:.3f}%; transform:translateX(-50%); '
            f'font-size:0.72rem; color:{color}; font-weight:{weight}; {underline}">{y}</div>'
        )
    st.markdown(
        f"""
        <div style="position:relative; height:18px; margin:-6px 6px 6px 6px;">
            {''.join(ticks_html)}
        </div>
        <div style="display:flex; gap:14px; font-size:0.78rem; color:#555; margin-bottom:6px;">
            <span><span style="color:{ACTUAL_TICK_COLOR}; font-weight:700;">&#9679;</span> actual year</span>
            <span><span style="color:{FORECAST_TICK_COLOR}; font-weight:700;">&#9679;</span> forecast year</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_clicked_fips(selection_state):
    """Pull FIPS out of a Streamlit plotly_chart on_select event payload
    (st.session_state[key] when the chart was built with
    on_select='rerun'). Returns None if nothing was clicked. Year is no
    longer read from the click event at all -- per the confirmed interface
    decision, the detail panel always shows the latest actual/observed
    year regardless of whether it was opened by clicking the map or by
    searching, so the click only needs to resolve which COUNTY was
    clicked."""
    if not selection_state:
        return None
    points = selection_state.get('selection', {}).get('points', [])
    if not points:
        return None
    cd = points[0].get('location') or points[0].get('customdata')
    if cd is None:
        return None
    if isinstance(cd, (list, tuple)):
        cd = cd[0]
    return str(cd).zfill(5)


def render_map_legend_note(no_data_note):
    """Small explicit legend strip under a map: a gray swatch + a short
    explanation, since Plotly's own colorbar can't show a 'no data' entry.
    Kept short and simple per the confirmed interface refinement (no
    verbose confidence-grade legend text)."""
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:8px; margin-top:4px;
                    margin-bottom:10px; font-size:0.85rem; color:#555;">
            <div style="width:14px; height:14px; background:{NO_DATA_COLOR};
                        border:1px solid rgba(80,80,80,0.5); flex-shrink:0;"></div>
            <span>{no_data_note}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
