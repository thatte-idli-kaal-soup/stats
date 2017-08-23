import numpy as np
from plotly.graph_objs import (
    Scatter, Data, Figure, Line, Layout, XAxis, YAxis, Margin, Marker
)

PI = np.pi


def moduloAB(x, a, b):
    # maps a real number onto the unit circle identified with
    # the interval [a,b), b-a=2*PI
    if a >= b:
        raise ValueError('Incorrect interval ends')
    y = (x-a) % (b-a)
    return y+b if y < 0 else y+a


def test_2PI(x):
    return 0 <= x < 2*PI


def get_ideogram_ends(ideogram_len, gap):
    ideo_ends = []
    left = 0
    for k in range(len(ideogram_len)):
        right = left+ideogram_len[k]
        ideo_ends.append([left, right])
        left = right+gap
    return ideo_ends


def make_ideogram_arc(R, phi, a=50):
    # R is the circle radius
    # phi is the list of ends angle coordinates of an arc
    # a is a parameter that controls the number of points to be evaluated on an arc
    if not test_2PI(phi[0]) or not test_2PI(phi[1]):
        phi = [moduloAB(t, 0, 2*PI) for t in phi]
    length = (phi[1]-phi[0]) % 2*PI
    nr = 5 if length <= PI/4 else int(a*length/PI)

    if phi[0] < phi[1]:
        theta = np.linspace(phi[0], phi[1], nr)

    else:
        phi = [moduloAB(t, -PI, PI) for t in phi]
        theta = np.linspace(phi[0], phi[1], nr)

    return R*np.exp(1j*theta)


def map_data(data_matrix, row_value, ideogram_length):
    mapped = np.zeros(data_matrix.shape)
    for j in range(len(data_matrix)):
        mapped[:, j] = ideogram_length*data_matrix[:, j]/row_value
    return mapped


def make_ribbon_ends(mapped_data, ideo_ends,  idx_sort):
    L = mapped_data.shape[0]
    ribbon_boundary = np.zeros((L, L+1))
    for k in range(L):
        start = ideo_ends[k][0]
        ribbon_boundary[k][0] = start
        for j in range(1, L+1):
            J = idx_sort[k][j-1]
            ribbon_boundary[k][j] = start+mapped_data[k][J]
            start = ribbon_boundary[k][j]
    return [[(ribbon_boundary[k][j],
              ribbon_boundary[k][j+1])
             for j in range(L)] for k in range(L)]


def control_pts(angle, radius):
    # angle is a 3-list containing angular coordinates of the control points
    # b0, b1, b2 radius is the distance from b1 to the origin O(0,0)
    b_cplx = np.array([np.exp(1j*angle[k]) for k in range(3)])
    b_cplx[1] = radius*b_cplx[1]
    return list(zip(b_cplx.real, b_cplx.imag))


def ctrl_rib_chords(l, r, radius):
    # this function returns a 2-list containing control poligons of the two
    # quadratic Bezier curves that are opposite sides in a ribbon l (r) the
    # list of angular variables of the ribbon arc ends defining the ribbon
    # starting (ending) arc radius is a common parameter for both control
    # polygons
    if len(l) != 2 or len(r) != 2:
        raise ValueError('the arc ends must be elements in a list of len 2')
    return [control_pts([l[j], (l[j]+r[j])/2, r[j]], radius) for j in range(2)]


def check_data(data_matrix):
    L, M = data_matrix.shape
    if L != M:
        raise ValueError('Data array must have (n,n) shape')
    return L


def make_ideo_shape(path, line_color, fill_color):
    # line_color is the color of the shape boundary
    # fill_collor is the color assigned to an ideogram
    return dict(
        line=Line(
            color=line_color,
            width=0.45
        ),

        path=path,
        type='path',
        fillcolor=fill_color,
    )


def make_layout(title, plot_size):
    # hide axis line, grid, tick-labels and title
    axis = dict(showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title='')

    return Layout(
        title=title,
        xaxis=XAxis(axis),
        yaxis=YAxis(axis),
        showlegend=False,
        width=plot_size,
        height=plot_size,
        margin=Margin(t=50, b=50, l=50, r=50),
        hovermode='closest',
        shapes=[]  # to this list one appends below the dicts defining the ribbon,
        # respectively the ideogram shapes
    )


def make_ribbon(l, r, line_color, fill_color, radius=0.2):
    # l=[l[0], l[1]], r=[r[0], r[1]]  represent the opposite arcs in the ribbon
    # line_color is the color of the shape boundary
    # fill_color is the fill color for the ribbon shape
    b, c = ctrl_rib_chords(l, r, radius)
    return dict(
        line=Line(color=line_color, width=0.5),
        path=(
            make_q_bezier(b) +
            make_ribbon_arc(r[0], r[1]) +
            make_q_bezier(c[::-1]) +
            make_ribbon_arc(l[1], l[0])
        ),
        type='path',
        fillcolor=fill_color,
    )


def make_self_rel(l, line_color, fill_color, radius):
    # radius is the radius of Bezier control point b_1
    b = control_pts([l[0], (l[0]+l[1])/2, l[1]], radius)
    return dict(
        line=Line(color=line_color, width=0.5),
        path=make_q_bezier(b)+make_ribbon_arc(l[1], l[0]),
        type='path',
        fillcolor=fill_color,
    )


def invPerm(perm):
    # function that returns the inverse of a permutation, perm
    inv = [0] * len(perm)
    for i, s in enumerate(perm):
        inv[s] = i
    return inv


def make_q_bezier(b):
    # defines the Plotly SVG path for a quadratic Bezier curve defined by the
    # list of its control points
    A, B, C = b
    return 'M ' + str(A[0]) + ',' +str(A[1]) + ' ' + 'Q ' + \
        str(B[0]) + ', ' + str(B[1]) + ' ' + \
        str(C[0]) + ', ' + str(C[1])


def make_ribbon_arc(theta0, theta1):

    if not(test_2PI(theta0) and test_2PI(theta1)):
        if test_2PI(theta0):
            theta1 = theta0
        elif test_2PI(theta1):
            theta0 = theta1
        else:
            theta0 = theta1 = 2 * PI

    if theta0 < theta1:
        theta0 = moduloAB(theta0, -PI, PI)
        theta1 = moduloAB(theta1, -PI, PI)
        # if theta0 * theta1 > 0:
        #     raise ValueError('incorrect angle coordinates for ribbon')

    nr = int(40*(theta0-theta1)/PI)
    if nr <= 2:
        nr = 3

    theta = np.linspace(theta0, theta1, nr)
    pts = np.exp(1j*theta)  # points on arc in polar complex form

    string_arc = ''
    for k in range(len(theta)):
        string_arc += 'L '+str(pts.real[k])+', '+str(pts.imag[k])+' '

    return string_arc


def get_colors(n):
    # FIXME: Allow specifying a color scheme
    from plotly import colors
    colors = [
        c.replace('rgb', 'rgba').replace(')', ', 0.66)')
        for c in colors.DEFAULT_PLOTLY_COLORS
    ]
    return colors * (int(n/10) + 1)


def create_chord_diagram(data, labels):
    L = check_data(data)
    gap = 2*PI*0.001
    row_sum = data.sum(axis=1)
    ideogram_length = (2 * PI * np.asarray(row_sum)/data.sum()) - gap
    ideo_ends = get_ideogram_ends(ideogram_length, gap)

    mapped_data = map_data(data, row_sum, ideogram_length)
    idx_sort = np.argsort(mapped_data, axis=1)
    ribbon_ends = make_ribbon_ends(mapped_data, ideo_ends, idx_sort)
    colors = get_colors(L)
    ribbon_color = [L*[color] for color in colors]

    layout = make_layout('Successful passes between Players', 800)

    ideograms = []

    for k in range(L):
        z = make_ideogram_arc(1.1, ideo_ends[k])
        zi = make_ideogram_arc(1.0, ideo_ends[k])
        line = Line(color=colors[k], shape='spline', width=0.25)
        trace = Scatter(x=z.real,
                        y=z.imag,
                        mode='lines',
                        line=line,
                        text=labels[k]+'<br>'+'{:d}'.format(int(row_sum[k])),
                        hoverinfo='text')
        ideograms.append(trace)

        path = ' '.join('L{},{}'.format(x.real, x.imag) for x in z).replace('L', 'M', 1)
        path += ' '.join('L{},{}'.format(x.real, x.imag) for x in zi[::-1])
        layout['shapes'].append(
            make_ideo_shape(path, 'rgb(150,150,150)', colors[k])
        )
        n = int(len(z)/2)
        for i, text in enumerate([labels[k], int(row_sum[k])]):
            angle = np.mean(ideo_ends[k]) * 180 / PI
            text = text
            angle = ((180 - angle) if 75 < angle < 285 else -angle) if i == 0 else (90 - angle)
            factor = 1.1 if i == 0 else 0.96
            annotation = {
                'text': text,
                'x': z.real[n]*factor,
                'y': z.imag[n]*factor,
                'textangle': angle,
                'showarrow': False,
                'opacity': 0.7,
            }
            layout['annotations'].append(annotation)



    # FIXME: Values set by trial and error
    radii_sribb = [0.4, 0.30, 0.35, 0.39, 0.12]
    radii_sribb = radii_sribb * 4

    ribbon_info = []
    for k, sigma in enumerate(idx_sort):
        sigma_inv = invPerm(sigma)
        for j in range(k, L):
            if data[k][j] == 0 and data[j][k] == 0:
                continue
            eta = idx_sort[j]
            eta_inv = invPerm(eta)
            l = ribbon_ends[k][sigma_inv[j]]
            if np.any(np.isnan(l)):
                continue

            if j == k:
                shape = make_self_rel(
                    l, 'rgb(175,175,175)', colors[k], radii_sribb[k]
                )
                layout['shapes'].append(shape)
                z = 0.9*np.exp(1j*(l[0]+l[1])/2)
                text = '{:d} passes by {} to self'.format(int(data[k][k]), labels[k])
                marker = Marker(size=0.5, color=colors[k])
                trace = Scatter(x=z.real,
                                y=z.imag,
                                mode='markers',
                                marker=marker,
                                text=text,
                                hoverinfo='text')
                ribbon_info.append(trace)

            else:
                r = ribbon_ends[j][eta_inv[k]]
                if np.any(np.isnan(r)):
                    continue
                # *NOTE* Reverse arc ends, otherwise you get a twisted ribbon
                shape = make_ribbon(l, r[::-1], 'rgb(175,175,175)', ribbon_color[k][j])
                layout['shapes'].append(shape)

                zi = 0.9 * np.exp(1j * (l[0] + l[1]) / 2)
                texti = '{:d} passes from {} to {}'.format(int(data[k][j]), labels[k], labels[j])
                tracei = Scatter(x=[zi.real],
                                 y=[zi.imag],
                                 mode='markers',
                                 marker=Marker(size=0.5, color=ribbon_color[k][j]),
                                 text=texti,
                                 hoverinfo='text')
                ribbon_info.append(tracei)

                zf = 0.9 * np.exp(1j * (r[0] + r[1]) / 2)
                textf = '{:d} passes from {} to {}'.format(int(data[j][k]), labels[j], labels[k])
                tracef = Scatter(x=[zf.real],
                                 y=[zf.imag],
                                 mode='markers',
                                 marker=Marker(size=0.5, color=ribbon_color[k][j]),
                                 text=textf,
                                 hoverinfo='text')
                ribbon_info.append(tracef)

    data = Data(ideograms+ribbon_info)
    fig = Figure(data=data, layout=layout)
    return fig
