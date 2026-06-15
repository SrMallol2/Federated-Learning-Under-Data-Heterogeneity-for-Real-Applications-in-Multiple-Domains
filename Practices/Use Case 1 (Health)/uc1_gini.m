function uc1_gini()
% UC1_GINI  Two-panel Gini heterogeneity figure for UC1 (filtered vs unfiltered).
%
%   Reproduces the "Gini heterogeneity" plot from 002_UC1_FederatedAnalysis,
%   but in the same MATLAB house style as the other UC1 figures
%   (LaTeX serif fonts, log Dirichlet-alpha axis, dotted grid, boxed legend).
%
%   Left panel : Gini of the per-client positive rate   (label  heterogeneity)
%   Right panel: Gini of the per-client dataset size     (quantity heterogeneity)
%   Two series per panel: unfiltered vs filtered partition.
%
%   Gini = 0  -> perfectly equal across the 5 clients
%   Gini = 1  -> one client holds everything
%
%   Data are the exact summary values produced by compute_gini_df / the
%   heterogeneity summary in 002_UC1_FederatedAnalysis.ipynb (seed 42).
%
%   Output: figures/uc1_gini.pdf  and  figures/uc1_gini.png

    %% ---- locate output folder (next to this script) -----------------------
    here = fileparts(mfilename('fullpath'));
    if isempty(here), here = pwd; end
    outdir = fullfile(here, 'figures');
    if ~exist(outdir, 'dir'), mkdir(outdir); end

    %% ---- data (from 002_UC1_FederatedAnalysis, seed 42) -------------------
    alpha = [0.1 0.5 1.0 5.0 10.0];                 % Dirichlet sweep (x-axis)

    % Gini of per-client positive rate  (label heterogeneity)
    gl_unf  = [0.600265 0.595928 0.549867 0.371504 0.264817];
    gl_filt = [0.507357 0.294838 0.311278 0.371504 0.264817];

    % Gini of per-client dataset size   (quantity heterogeneity)
    gs_unf  = [0.746296 0.401387 0.363836 0.180184 0.128980];
    gs_filt = [0.485534 0.348679 0.426214 0.180184 0.128980];

    %% ---- style ------------------------------------------------------------
    cBlue = [0.2745 0.5098 0.7059];   % steelblue  -> unfiltered
    cRed  = [0.8863 0.2902 0.2000];   % #E24A33     -> filtered
    yMax  = 0.8;                      % shared y-limit (all Gini < 0.8)

    fig = figure('Color', 'w', 'Units', 'centimeters', ...
                 'Position', [2 2 24 9.5]);
    tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    % ---- Panel 1: label heterogeneity ----
    ax1 = nexttile(tl);
    panel(ax1, alpha, gl_unf, gl_filt, cBlue, cRed, yMax);
    ylabel(ax1, 'Gini (positive rate across clients)', 'Interpreter', 'latex');
    title(ax1, 'Label heterogeneity', 'Interpreter', 'latex', 'FontWeight', 'bold');

    % ---- Panel 2: quantity heterogeneity ----
    ax2 = nexttile(tl);
    [hUnf, hFilt] = panel(ax2, alpha, gs_unf, gs_filt, cBlue, cRed, yMax);
    ylabel(ax2, 'Gini (dataset size across clients)', 'Interpreter', 'latex');
    title(ax2, 'Quantity heterogeneity', 'Interpreter', 'latex', 'FontWeight', 'bold');

    % ---- shared legend along the bottom ----
    lg = legend([hUnf hFilt], ...
                {'Unfiltered', 'Filtered (actual experiment)'}, ...
                'Interpreter', 'latex', 'Orientation', 'horizontal', ...
                'FontSize', 11, 'Box', 'on');
    lg.Layout.Tile = 'south';

    %% ---- export -----------------------------------------------------------
    pdfPath = fullfile(outdir, 'uc1_gini.pdf');
    pngPath = fullfile(outdir, 'uc1_gini.png');
    exportgraphics(fig, pdfPath, 'ContentType', 'vector');
    exportgraphics(fig, pngPath, 'Resolution', 300);
    fprintf('Saved:\n  %s\n  %s\n', pdfPath, pngPath);
end

% ===========================================================================
function [hUnf, hFilt] = panel(ax, alpha, yUnf, yFilt, cBlue, cRed, yMax)
    hold(ax, 'on');
    hUnf  = plot(ax, alpha, yUnf,  'o-',  'Color', cBlue, ...
                 'MarkerFaceColor', cBlue, 'LineWidth', 2, 'MarkerSize', 7);
    hFilt = plot(ax, alpha, yFilt, 's--', 'Color', cRed, ...
                 'MarkerFaceColor', cRed,  'LineWidth', 2, 'MarkerSize', 7);
    hold(ax, 'off');

    set(ax, 'XScale', 'log', ...
            'TickLabelInterpreter', 'latex', 'FontSize', 11, ...
            'XGrid', 'on', 'YGrid', 'on', 'GridLineStyle', ':', ...
            'GridAlpha', 0.25, 'Box', 'on', 'Layer', 'top');
    xlim(ax, [0.085 12]);
    ylim(ax, [0 yMax]);
    xticks(ax, [0.1 0.5 1 5 10]);
    xticklabels(ax, {'0.1', '0.5', '1', '5', '10'});
    xlabel(ax, 'Dirichlet $\alpha$', 'Interpreter', 'latex');
end
