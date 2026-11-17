"""Ball location for every frame of the tactical video: observed where it was seen, filled where not.

    python -m ballfill.run       # -> cache/BAR-ATH_ball.csv (see README.md)

Modules
    data      broadcast parsing, sub-frame alignment to the tactical timeline, live-frame check
    fuse      one observed ball track from both cameras, with source and uncertainty per frame
    carrier   carrier classifier + HMM through a gap
    fill      gap-filling methods (baselines, oracle, carrier HMM)
    evaluate  scoring against real held-out truth (A: real gaps, B: synthetic gaps)
    tune      two-fold by-half parameter selection
    run       the full-match output

Everything derived from the match data is cached under cache/ (gitignored): never commit it.
"""

