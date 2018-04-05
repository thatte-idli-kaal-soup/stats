from collections import Counter, OrderedDict
import os
from os.path import abspath, dirname, exists, join

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.figure_factory as ff
import pandas

from chord_diagram import create_chord_diagram

HERE = dirname(abspath(__file__))
STATS_FILE = join(HERE, 'data', 'stats.csv')
TOURNAMENT = 'BUO 2017'
TEAM = 'TIKS'
markers = ['circle', 'square', 'cross', 'diamond']

if exists(STATS_FILE):
    DATA = pandas.read_csv(STATS_FILE)

else:
    from io import BytesIO
    from cryptography.fernet import Fernet
    key = os.environ.get('CRYPT_KEY', 'secret')
    with open('{}.crypt'.format(STATS_FILE), 'rb') as g:
        crypt_data = g.read()
    DATA = pandas.read_csv(BytesIO(Fernet(key).decrypt(crypt_data)))
OPPONENTS = sorted(DATA[DATA['Tournament'] == TOURNAMENT]['Opponent'].unique())


def get_match_events(tournament, opponent):
    points = DATA[(DATA['Tournament'] == tournament) &
                  (DATA['Opponent'] == opponent)]
    # FIXME: Disambiguate by date if there are multiple matches!
    assert len(points['Date/Time'].unique()) == 1, 'Multiple matches!'
    return points


def get_score_line(points):
    goals = points[points['Action'] == 'Goal']
    return (
        goals['Their Score - End of Point'],
        goals['Our Score - End of Point']
    )


def score_line_figure():
    traces = []
    for i, opponent in enumerate(OPPONENTS):
        events = get_match_events(TOURNAMENT, opponent)
        theirs, ours = get_score_line(events)
        traces.append({
            'x': ours,
            'y': theirs,
            'name': opponent,
            'line': {'dash': 'dashdot', },
            'marker': {'symbol': markers[i % 4], 'size': 8},
            'mode': 'lines+markers',
            'legendgroup': opponent,
        })
        traces.append({
            'x': list(range(16)),
            'y': list(range(16)),
            'name': 'Scores level',
            'mode': 'lines',
            'opacity': 0.5,
            'legendgroup': opponent,
            'showlegend': False,
            'hoverinfo': "none",
            'line': {
                'color': '#666',
                'width': 0.5,
            }
        })

    figure = {
        'data': traces,
        'layout': {
            'height': 800,
            'width': 800,
            'hovermode': 'closest',
            'xaxis': {
                'title': 'Our score',
            },
            'yaxis': {
                'title': 'Opponent score',
            },
            'font': {
                'color': 'black'
            }
        }
    }

    return figure


def player_o_d_stats():
    columns = ['Player {}'.format(i) for i in range(7)]
    players = set(list(DATA[columns].fillna('V').values.flatten()))
    players.remove('V')
    players = sorted(players)
    goals = DATA[(DATA['Tournament'] == TOURNAMENT) &
                 (DATA['Action'] == 'Goal')]

    player_stats = {}
    for player in players:
        points = goals[(goals[columns] == player).any(axis=1)]
        o_points = points[points['Line'] == 'O']
        if len(points) < 5:
            continue
        player_stats[player] = (
            len(points), len(o_points), len(points) - len(o_points)
        )

    return player_stats


def o_d_lines_figure():
    player_stats = player_o_d_stats()
    scatter_data = {
        'x': [o for _, o, _ in player_stats.values()],
        'y': [d for _, _, d in player_stats.values()],
        'text': ['{} - {}'.format(name, t)
                 for name, (t, o, d) in player_stats.items()],
        'mode': 'markers+text',
        'hoverinfo': 'x+y+text',
        'showlegend': False,
        'marker': {
            'size': [t for t, _, _ in player_stats.values()],
            'color': [t/5 for t, _, _ in player_stats.values()],
        }
    }

    traces = [go.Scatter(scatter_data)]
    traces.append({
        'x': list(range(60)),
        'y': list(range(60)),
        'name': 'O-D split',
        'mode': 'lines',
        'showlegend': False,
    })

    figure = {
        'data': traces,
        'layout': {
            'height': 800,
            'width': 900,
            'hovermode': 'closest',
            'xaxis': {
                'title': 'Offense',
            },
            'yaxis': {
                'title': 'Defense',
            },
        }
    }

    return figure


def time_between_points():
    table = []
    columns = ['Opponent', 'Median Time (secs)', 'Actual times (secs)']
    for opponent in OPPONENTS:
        events = get_match_events(TOURNAMENT, opponent)
        our_goals = events[(events['Action'] == 'Goal') &
                           (events['Event Type'] == 'Offense')]
        pull_times = []
        for row_index in our_goals.index:
            goal = DATA.loc[row_index]
            next_event = DATA.loc[row_index+1]
            # Ignore if match changed, or half-time
            if (
                    next_event['Opponent'] != opponent or
                    next_event['Event Type'] != 'Defense' or
                    not next_event['Action'].startswith('Pull')  # Pull(Ob)
            ):
                continue
            pull_time = (
                next_event['Elapsed Time (secs)'] - goal['Elapsed Time (secs)']
            )
            pull_times.append(pull_time)
        table.append([
            opponent,
            int(pandas.Series(pull_times).median()),
            ', '.join(map(str, pull_times))
        ])
    table = ff.create_table(pandas.DataFrame(table, columns=columns),)
    # FIXME: This is a plot.ly bug - headings are not visible, by default
    for annotation in table['layout']['annotations']:
        annotation['font']['color'] = '#000000'
    return table


def passes_histogram(chosen_opponent=None):
    opponent_data = {}
    for opponent in OPPONENTS:
        events = get_match_events(TOURNAMENT, opponent)
        goals = events[(events['Event Type'] == 'Offense') &
                       (events['Action'] == 'Goal')]
        passes = events[(events['Event Type'] == 'Offense') &
                         (events['Action'] == 'Catch')]
        drops = events[(events['Event Type'] == 'Offense') &
                       (events['Action'] == 'Drop')]
        throwaways = events[
            (events['Event Type'] == 'Offense') &
            (events['Action'] == 'Throwaway')
        ]

        # Possession gaining events -
        # defense-goal, defense-throwaway, defense-d
        possession_change_events = events[
            (events['Event Type'] == 'Defense') &
            (
                (events['Action'] == 'Goal') |
                (events['Action'] == 'Throwaway') |
                (events['Action'] == 'D')
            )
        ]

        def passes_since_possession(event_index):
            indexes = possession_change_events.index[
                possession_change_events.index < event_index
            ]
            possession = (
                indexes[-1] if len(indexes) > 0 else (events.index[0] - 1)
            )
            return event_index - possession

        opponent_data[opponent] = {
            'goals': goals.index.map(passes_since_possession),
            'passes': passes.index.map(passes_since_possession),
            'drops': drops.index.map(passes_since_possession),
            'throwaways': throwaways.index.map(passes_since_possession),
        }

    if chosen_opponent:
        data = opponent_data[chosen_opponent]
        goals = data['goals']
        passes = data['passes']
        drops = data['drops']
        throwaways = data['throwaways']

    else:
        data = opponent_data.values()
        goals = [
            count
            for value in data
            for count in value['goals']
        ]
        passes = [
            count
            for value in data
            for count in value['passes']
        ]
        drops = [
            count
            for value in data
            for count in value['drops']
        ]
        throwaways = [
            count
            for value in data
            for count in value['throwaways']
        ]

    goals = Counter(goals)
    passes = Counter(passes)
    drops = Counter(drops)
    throwaways = Counter(throwaways)

    turnovers = Counter()
    turnovers.update(drops)
    turnovers.update(throwaways)

    catches = Counter()
    catches.update(passes)
    catches.update(goals)

    data = [
        go.Bar(
            x=list(goals.keys()),
            y=list(goals.values()),
            name='Goals',
        ),
        go.Bar(
            x=list(passes.keys()),
            y=list(passes.values()),
            name='Passes',
        ),
        go.Bar(
            x=list(catches.keys()),
            y=list(catches.values()),
            name='Catches',
        ),
        go.Bar(
            x=list(drops.keys()),
            y=list(drops.values()),
            name='Drops',
        ),
        go.Bar(
            x=list(throwaways.keys()),
            y=list(throwaways.values()),
            name='Throwaways',
        ),
        go.Bar(
            x=list(turnovers.keys()),
            y=list(turnovers.values()),
            name='Turnovers',
        ),
    ]

    figure = {
        'data': data,
        'layout': {
            'xaxis': {
                'title': 'Pass # for event',
                'dtick': 1,
            },
            'yaxis': {
                'title': '# of occurrences',
            }

        },
    }

    return figure


def time_between_points_div():
    return html.P(
        "We look at median time (and not mean time) to compensate for not"
        " knowing how accurate the data is -- timeouts, data entry delay, etc."
        " can add outliers"
    )


def match_events():
    figures = []
    for i, opponent in enumerate(OPPONENTS):
        events = get_match_events(TOURNAMENT, opponent)
        offense = events[(events['Event Type'] == 'Offense')]
        defense = events[(events['Event Type'] == 'Defense')]
        match_data = {
            'Catch': offense[offense['Action'] == 'Catch']['Elapsed Time (secs)'].values,
            'Drop': offense[offense['Action'] == 'Drop']['Elapsed Time (secs)'].values,
            'Throwaway': offense[offense['Action'] == 'Throwaway']['Elapsed Time (secs)'].values,
            'Goal': offense[offense['Action'] == 'Goal']['Elapsed Time (secs)'].values,
            'Opponent Goal': defense[defense['Action'] == 'Goal']['Elapsed Time (secs)'].values,
        }

        figure = ff.create_distplot(
            hist_data=list(match_data.values()),
            group_labels=list(match_data.keys()),
            curve_type='normal',
            show_rug=True,
            show_curve=False,
            show_hist=False,
        )

        markers = {
            'Throwaway': 'pentagon-open',
            'Drop': 'hexagon-open',
            'Catch': 'line-ns-open',
            'Goal': 'star',
            'Opponent Goal': 'star-open',
        }
        # Change the markers!
        for trace in figure['data']:
            trace['marker']['symbol'] = markers[trace['name']]
            trace['y'] = [opponent] * len(trace['y'])
            if trace['name'] == 'Catch':
                trace['marker']['size'] = 10
            if i > 0:
                trace['showlegend'] = False

        figure['layout'].pop('yaxis1')
        figure['layout'].pop('yaxis2')
        figures.append(figure)

    figure = {
        'data': [trace for fig in figures for trace in fig['data']],
        'layout': figures[0]['layout'],
    }

    return figure


def passes_chord_diagram():
    events = DATA[DATA['Tournament'] == TOURNAMENT]
    successful_passes = events[
        ((events['Action'] == 'Goal') | (events['Action'] == 'Catch')) &
        events['Passer'].notnull() &
        (events['Passer'] != 'Anonymous') &
        events['Receiver'].notnull()
    ]
    passes = pandas.pivot_table(
        successful_passes[['Passer', 'Receiver']],
        index=['Passer'],
        columns=['Receiver'],
        aggfunc=len
    )

    rows = set(passes.index)
    for player in passes.columns:
        if player in rows:
            continue
        s = pandas.Series(
            [pandas.np.nan]*len(passes.columns),
            index=passes.columns,
            name=player
        )
        passes = passes.append(s).sort_index()

    player_stats = player_o_d_stats()
    # Order by no. of offensive points played
    order_by = sorted(player_stats,
                      key=lambda x: player_stats[x][1],
                      reverse=True)
    data = passes.loc[order_by][order_by].fillna(0).values

    return create_chord_diagram(data, order_by)

graph_types = OrderedDict(
    [
        ('Score Line', score_line_figure),
        ('O-D Lines', o_d_lines_figure),
        ('Time between points', time_between_points),
        ('Passes histogram', passes_histogram),
        ('Match events', match_events),
        ('Passes between players', passes_chord_diagram),
    ]
)

div_types = OrderedDict(
    [('Time between points', time_between_points_div), ]
)


def create_app(server, url_base_pathname):
    app = dash.Dash(TOURNAMENT, server=server,
                    url_base_pathname=url_base_pathname)
    app.title = '{} Statistics'.format(TOURNAMENT)

    app.scripts.config.serve_locally = False
    dcc._js_dist[0]['external_url'] = (
        'https://cdn.plot.ly/plotly-basic-latest.min.js'
    )

    app.layout = html.Div([
        html.H1('{} - {}'.format(TOURNAMENT, TEAM)),
        dcc.Dropdown(
            id='graph-type-dropdown',
            options=[{'label': graph_type, 'value': graph_type}
                     for graph_type in list(graph_types.keys())],
            value='Passes between players',
        ),
        dcc.Dropdown(
            id='opponent-dropdown',
            options=[{'label': opponent, 'value': opponent}
                     for opponent in OPPONENTS] + [{'label': 'All', 'value': None}],
            placeholder='Opponent...',
            value=None
        ),
        dcc.Graph(id='my-graph'),
        html.Div(id='my-div')
    ], className="container")

    app.css.append_css({
        'external_url': "https://codepen.io/chriddyp/pen/bWLwgP.css"
    })

    @app.callback(Output('my-graph', 'figure'),
                  [Input('graph-type-dropdown', 'value'),
                   Input('opponent-dropdown', 'value')])
    def update_graph(graph_type, opponent):
        if graph_type not in graph_types:
            return None
        f = graph_types[graph_type]
        return f() if f.__code__.co_argcount == 0 else f(opponent)

    @app.callback(Output('opponent-dropdown', 'disabled'),
                  [Input('graph-type-dropdown', 'value')])
    def show_or_hide_opponent_dropdown(graph_type):
        f = graph_types[graph_type]
        return True if f.__code__.co_argcount == 0 else False

    @app.callback(Output('my-div', 'children'),
                  [Input('graph-type-dropdown', 'value'), ])
    def update_div(div_type):
        return div_types[div_type]() if div_type in div_types else None

    return app
