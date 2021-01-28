NeoPlots
========

.. code:: python

    import NeoPlots
    import neo
    from neo.test.generate_datasets import generate_one_simple_segment

    import matplotlib.colors as mcolors
    colors = list(mcolors.TABLEAU_COLORS.values())[1:]

    # generate some random data and add annotations
    seg = generate_one_simple_segment(supported_objects=neo.objectlist)
    for i, sig in enumerate(seg.analogsignals):
        sig.name = f'sig {i}'
    for i, st in enumerate(seg.spiketrains):
        st.annotate(channels=[f'sig {i % len(seg.analogsignals)}'])
        st.annotate(color=colors[i % len(colors)])
    for i, ep in enumerate(seg.epochs):
        ep.durations *= 0.1  # reduce durs so epochs are easier to see in this random example
        ep.annotate(color=colors[i % len(colors)])

    # generate, save, and display the plot
    engine = 'matplotlib'  # or 'plotly'
    fig = NeoPlots.plot_segment(engine=engine, segment=seg,
                                ylims={sig.name: (-0.5, 1.5) for sig in seg.analogsignals})
    NeoPlots.save_fig(fig, 'myfig')
    fig.show()
