"""

"""

import numpy as np
import neo

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False

try:
    import plotly
    from plotly.subplots import make_subplots
    HAVE_PLOTLY = True
except ImportError:
    HAVE_PLOTLY = False


def get_amplitudes(sig, times):
    return sig.as_quantity()[sig.time_index(times)]


def plot_segment(*args, **kwargs):
    seg = kwargs.pop('segment', None)
    if isinstance(seg, neo.Segment):
        sigs = kwargs.pop('sigs', seg.analogsignals)
        spiketrains = kwargs.pop('spiketrains', seg.spiketrains)
        epochs = kwargs.pop('epochs', seg.epochs)
        title = kwargs.pop('title', None)
        fig = plot_signals(*args, sigs=sigs, spiketrains=spiketrains, **kwargs)
        add_epochs_to_fig(fig, epochs)
        add_title_to_fig(fig, title)
        return fig
    else:
        raise ValueError(f"a neo.Segment must be provided with the segment parameter")


def plot_signals(*args, **kwargs):
    engine = kwargs.pop('engine', None)
    if engine == 'matplotlib':
        return plot_signals_matplotlib(*args, **kwargs)
    elif engine == 'plotly':
        return plot_signals_plotly(*args, **kwargs)
    else:
        raise ValueError(f"engine='matplotlib' or engine='plotly' is required")


# plot using matplotlib (non-interactive, but unlimited by data size)
def plot_signals_matplotlib(sigs, spiketrains=[], new_units={}, ylims={}, t_start=None, t_stop=None, fig_width=None, fig_height=None, dpi=72):

    # ensure sigs is a 2D list of single-channel AnalogSignals
    # - each row in the 2D list corresponds to a panel in the figure
    # - each signal within a row will be plotted within that panel
    if isinstance(sigs, neo.AnalogSignal):
        sigs = [[sigs]]
    elif isinstance(sigs, list) and isinstance(sigs[0], neo.AnalogSignal):
        sigs = [[sig] for sig in sigs]
    elif isinstance(sigs, dict):
        sigs = [[sig] for sig in sigs.values()]
    for row in sigs:
        for sig in row:
            if not isinstance(sig, neo.AnalogSignal):
                raise ValueError(f'non-AnalogSignal found in sigs: {sig}')
            elif sig.shape[1] != 1:
                raise ValueError(f'AnalogSignal must be single-channel: {sig}')

    # ensure new_units is a dictionary
    if isinstance(new_units, str):
        # prepare to use this one unit type for all signals
        kvpairs = []
        for row in sigs:
            for sig in row:
                kvpairs.append((sig.name, new_units))
        new_units = dict(kvpairs)
    elif not isinstance(new_units, dict):
        raise ValueError(f'new_units must either be a string or dict: {new_units}')

    # generate an empty multi-panel matplotlib figure object
    # - nrows determines the number of vertically stacked panels
    # - shared_x ensures that x-axes are synced
    # - figsize determines the width and height in inches
    fig_width_inches = fig_width / dpi if fig_width else 18
    fig_height_inches = fig_height / dpi if fig_height else 5
    fig, axes = plt.subplots(nrows=len(sigs), ncols=1, squeeze=False, sharex=True, figsize=(fig_width_inches, fig_height_inches))

    # iterate over all AnalogSignals
    for i, panel_sigs in enumerate(sigs):
        ax = axes[i][0]
        for j, sig in enumerate(panel_sigs):

            # AnalogSignal.time_slice() extracts the section of the signal between
            # t_start and t_stop
            # - if t_start or t_stop is None, time_slice will not clip the
            #   beginning or end of the signal, respectively
            sig = sig.time_slice(t_start, t_stop)

            # TODO: add comments to the rest of this......

            color = sig.annotations.get('color', None)

            x = sig.times
            y = sig.as_quantity().rescale(new_units.get(sig.name, sig.units))
            ax.plot(x, y, c=color, label=sig.name)

            ax.set_xlim(sig.t_start, sig.t_stop)
            if sig.name in ylims:
                ax.set_ylim(ylims[sig.name][0], ylims[sig.name][1])

            ax.grid(True)

            if j == 0:
                # set y-axis label to first signal's name
                ax.set_ylabel(f'{sig.name} ({y.dimensionality})')

    # update last x-axis label
    ax.set_xlabel(f'Time ({x.dimensionality})')

    # plot spiketrains as points
    if spiketrains:
        for st in spiketrains:
            st = st.time_slice(t_start, t_stop)
            channel = st.annotations['channels'][0]
            color = st.annotations.get('color', None)
            panel_index = None
            for i, row in enumerate(sigs):
                for sig in row:
                    if sig.name == channel:
                        panel_index = i
                        break
                if panel_index is not None:
                    break
            x = st.times
            y = get_amplitudes(sig, x).rescale(new_units.get(sig.name, sig.units)).flatten()
            ax = axes[panel_index][0]
            ax.scatter(x, y, zorder=3, c=color, label=st.name)

    fig.tight_layout()

    return fig


# plot using plotly (interactive, but cannot handle more than a few million points)
def plot_signals_plotly(sigs, spiketrains=[], new_units={}, ylims={}, t_start=None, t_stop=None, fig_width=None, fig_height=None, horizontal_spacing=None, vertical_spacing=None, downsample_threshold=1e6):

    # ensure sigs is a 2D list of single-channel AnalogSignals
    # - each row in the 2D list corresponds to a panel in the figure
    # - each signal within a row will be plotted within that panel
    if isinstance(sigs, neo.AnalogSignal):
        sigs = [[sigs]]
    elif isinstance(sigs, list) and isinstance(sigs[0], neo.AnalogSignal):
        sigs = [[sig] for sig in sigs]
    elif isinstance(sigs, dict):
        sigs = [[sig] for sig in sigs.values()]
    for row in sigs:
        for sig in row:
            if not isinstance(sig, neo.AnalogSignal):
                raise ValueError(f'non-AnalogSignal found in sigs: {sig}')
            elif sig.shape[1] != 1:
                raise ValueError(f'AnalogSignal must be single-channel: {sig}')

    # ensure new_units is a dictionary
    if isinstance(new_units, str):
        # prepare to use this one unit type for all signals
        kvpairs = []
        for row in sigs:
            for sig in row:
                kvpairs.append((sig.name, new_units))
        new_units = dict(kvpairs)
    elif not isinstance(new_units, dict):
        raise ValueError(f'new_units must either be a string or dict: {new_units}')

    # generate an empty multi-panel Plotly figure object
    # - rows determines the number of vertically stacked panels
    # - shared_xaxes ensures that panning and zooming in time in one panel
    #   updates all
    # - horizontal_spacing and vertical_spacing must be None or numbers between
    #   0 and 1
    fig = make_subplots(rows=len(sigs), shared_xaxes=True,
                        horizontal_spacing=horizontal_spacing,
                        vertical_spacing=vertical_spacing)

    # get signal amplitudes at spike times
    # - this is done before signal downsampling so that the original amplitudes
    #   are fetched accurately
    for st in spiketrains:
        channel = st.annotations['channels'][0]
        sig_found = False
        for i, row in enumerate(sigs):
            for sig in row:
                if sig.name == channel:
                    sig_found = True
                    break
            if sig_found:
                break
        assert sig.name == channel
        spike_amplitudes = get_amplitudes(sig, st.times)
        st.array_annotate(spike_amplitudes=spike_amplitudes)

    # reduce the number of data points until it is below downsample_threshold
    # - with more than a few million data points at most (sometimes less),
    #   Colab will fail to display the figure, reporting "Runtime disconnected"
    if downsample_threshold is not None:
        n_points = sum([sum([sig.time_slice(t_start, t_stop).size for sig in row]) for row in sigs])
        if n_points > downsample_threshold:
            downsample_factor = 2
            while np.ceil(n_points / downsample_factor) + 1 > downsample_threshold:
                downsample_factor += 1
            for i, row in enumerate(sigs):
                for j, sig in enumerate(row):
                    sigs[i][j] = sig[::downsample_factor]
                    sigs[i][j].sampling_period *= downsample_factor
            print(f"Downsampled total points from {n_points} to {sum([sum([sig.time_slice(t_start, t_stop).size for sig in row]) for row in sigs])} using a downsample factor of {downsample_factor}.")
            print(f"The effective sample rate for plotting will be {sigs[0][0].sampling_rate:g} ({sigs[0][0].sampling_period.rescale('ms'):g}).")  # assuming equal rates for all sigs

    # iterate over all AnalogSignals
    for i, row in enumerate(sigs):
        for j, sig in enumerate(row):

            # AnalogSignal.time_slice() extracts the section of the signal between
            # t_start and t_stop
            # - if t_start or t_stop is None, time_slice will not clip the
            #   beginning or end of the signal, respectively
            sig = sig.time_slice(t_start, t_stop)

            # when creating a scatter or line plot using Plotly with regularly
            # sampled points, rather than passing an array of x-values (like
            # sig.times) it is more effecient to pass a starting x-value (x0) and
            # the spacing between x-values (dx)
            # - float() converts these Quantities to simple floating point numbers,
            #   since Plotly sometimes struggles with Quantity objects
            x0 = float(sig.t_start)
            dx = float(sig.sampling_period.rescale(sig.t_start.units))

            # obtain y-values by rescaling to new units if any were provided
            # (otherwise convert to current units, which does nothing) and
            # flattening the N-by-1 matrix (technically 2-dimensional) into a
            # 1-dimensional vector
            y = sig.rescale(new_units.get(sig.name, sig.units)).flatten()

            # TODO add comment
            color = sig.annotations.get('color', None)

            # plot the data in the current figure panel
            fig.add_scatter(x0=x0, dx=dx, y=y, line_color=color, name=sig.name, row=i+1, col=1)

            # update the x-axis range in the current figure panel
            fig.update_xaxes(range=(sig.t_start, sig.t_stop), row=i+1)

            # update the y-axis label and range in the current figure panel for
            # the first signal only
            if j == 0:
                fig.update_yaxes(title_text=f'{sig.name} ({y.dimensionality})', range=ylims.get(sig.name, None), row=i+1)

    # update last x-axis label
    fig.update_xaxes(title_text=f'Time ({sig.times.dimensionality})', row=len(sigs))

    # plot spiketrains as points
    if spiketrains:
        for st in spiketrains:
            st = st.time_slice(t_start, t_stop)
            channel = st.annotations['channels'][0]
            color = st.annotations.get('color', None)
            panel_index = None
            for i, row in enumerate(sigs):
                for sig in row:
                    if sig.name == channel:
                        panel_index = i
                        break
                if panel_index is not None:
                    break
            x = st.times
            y = st.array_annotations['spike_amplitudes'].rescale(new_units.get(sig.name, sig.units)).flatten()
            fig.add_scatter(x=x, y=y, mode='markers', marker_color=color, name=st.name, row=panel_index+1, col=1)

    # remove the unnecessary legend and adjust the height of the entire figure
    fig.update_layout(showlegend=False, width=fig_width, height=fig_height)

    return fig


def get_engine_from_fig(fig):
    if isinstance(fig, plt.Figure):
        return 'matplotlib'
    elif isinstance(fig, plotly.graph_objs.Figure):
        return 'plotly'
    else:
        raise ValueError(f'fig has unrecognized type: {type(fig)}')


def add_epochs_to_fig(fig, epochs):
    '''Plot epochs as rectangles behind traces across all subplot panels'''

    engine = get_engine_from_fig(fig)

    if epochs is None:
        return fig

    for ep in epochs:
        color = ep.annotations.get('color', '#AAAAAA')
        for ep_start, ep_duration in zip(ep.times, ep.durations):
            ep_stop = ep_start + ep_duration

            if engine == 'matplotlib':
                for ax in fig.get_axes():
                    x_min, x_max, y_min, y_max = ax.axis()
                    ax.add_patch(patches.Rectangle(
                        (ep_start, y_min), ep_duration, y_max - y_min,
                        facecolor=color, alpha=0.2, zorder=-1))
                    ax.set_xlim(x_min, x_max)

            elif engine == 'plotly':
                fig.add_shape(
                    type='rect', x0=float(ep_start), x1=float(ep_stop),
                    fillcolor=color, line=dict(width=0), opacity=0.2, layer='below',
                    ysizemode='scaled', yref='paper', y0=0, y1=1)

    return fig


def add_title_to_fig(fig, title):
    '''Add a title to the figure'''

    engine = get_engine_from_fig(fig)

    if title is None:
        return fig

    if engine == 'matplotlib':
        fig.axes[0].set_title(title)

    elif engine == 'plotly':
        fig.update_layout(title=title)

    return fig


def save_fig(fig, basename):
    '''Save the figure to a file'''

    engine = get_engine_from_fig(fig)

    if engine == 'matplotlib':
        fig.savefig(basename + '.png')
    elif engine == 'plotly':
        with open(basename + '.html', 'w') as f:
            f.write(fig.to_html())
