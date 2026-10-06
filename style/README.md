# style

`axisforge.mplstyle` — the single report style of the repository: ISO 80000-1
axis labels (`$v$ / mm`), series identity never carried by colour alone.

Validation cases use it through `validation._support.style.use_style()`.
Studies and examples use the file directly: `plt.style.use("<path>/style/axisforge.mplstyle")`.
