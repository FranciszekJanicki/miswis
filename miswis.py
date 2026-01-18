import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State


class PID:
    def __init__(self, Kp, Ti, Td, windup=100000000, offset=0):
        self.Kp = Kp
        self.Ki = Kp/Ti
        self.Kd = Kp*Td
        self.windup = windup
        self.offset = offset
        self.integral = 0
        self.prev_t = 0
        self.prev_e = 0

    def step(self, curr_t, curr_e):
        delta_t = curr_t - self.prev_t
        delta_e = curr_e - self.prev_e

        proportion = self.Kp * curr_e
        self.integral = self.integral + self.Ki * curr_e * delta_t

        if (self.integral > self.windup):
            self.integral = self.windup + self.offset
        elif (self.integral < -self.windup):
            self.integral = -self.windup - self.offset

        if (abs(delta_t) < 0.001):
            derivative = 0
        else:
            derivative = self.Kd * delta_e / delta_t

        u_signal = proportion + self.integral + derivative + self.offset

        self.prev_e = curr_e
        self.prev_t = curr_t
        return u_signal


def equations(state, t, params, pid_controller):
    theta, dtheta, phi, dphi, x, dx, px, py = state
    IB, IV, R, b, LT, mB, mW, g, IW = params

    val = pid_controller.step(t, theta)
    uL = val + 4
    uR = val

    dpsiR = (dx - (b * dphi) / 2) / R
    dpsiL = (dx + (b * dphi) / 2) / R

    TL = (uL - dpsiL) * 10000
    TR = (uR - dpsiR) * 10000

    ddtheta = ((2 * IW * LT * g * mB * np.sin(theta) + IW * LT ** 2 * dphi ** 2 * mB * np.sin(
        2 * theta) - 12 * LT ** 6 * R ** 2 * dtheta ** 6 * mB ** 2 * np.cos(theta) * np.sin(
        theta) - 6 * LT ** 3 * R ** 3 * TL * dtheta ** 2 * mB * np.cos(
        theta) - 6 * LT ** 3 * R ** 3 * TR * dtheta ** 2 * mB * np.cos(
        theta) + 6 * LT * R ** 2 * dx ** 2 * g * mB ** 2 * np.sin(
        theta) - 8 * LT ** 2 * R ** 3 * TL * dtheta * dx * mB - 8 * LT ** 2 * R ** 3 * TR * dtheta * dx * mB + LT ** 4 * R ** 2 * dphi ** 2 * dtheta ** 2 * mB ** 2 * np.sin(
        2 * theta) + 3 * LT ** 2 * R ** 2 * dphi ** 2 * dx ** 2 * mB ** 2 * np.sin(
        2 * theta) + 12 * LT ** 2 * R ** 2 * dtheta ** 2 * dx ** 4 * mB ** 2 * np.sin(
        2 * theta) - 28 * LT ** 4 * R ** 2 * dtheta ** 4 * dx ** 2 * mB ** 2 * np.sin(
        2 * theta) + 2 * LT * R ** 2 * g * mB * mW * np.sin(
        theta) + 4 * IW * LT ** 2 * dtheta ** 2 * dx ** 2 * mB * np.sin(
        2 * theta) - 8 * LT ** 5 * R ** 2 * dtheta ** 5 * dx * mB ** 2 * np.sin(
        theta) + 2 * LT ** 3 * R ** 2 * dtheta ** 2 * g * mB ** 2 * np.sin(
        theta) + LT ** 2 * R ** 2 * dphi ** 2 * mB * mW * np.sin(
        2 * theta) - 6 * LT * R ** 3 * TL * dx ** 2 * mB * np.cos(theta) - 6 * LT * R ** 3 * TR * dx ** 2 * mB * np.cos(
        theta) + 8 * IW * LT ** 3 * dtheta ** 3 * dx * mB * np.sin(
        theta) - 24 * LT ** 3 * R ** 2 * dtheta ** 3 * dx ** 3 * mB ** 2 * np.sin(
        theta) - 24 * LT ** 3 * R ** 2 * dtheta ** 3 * dx ** 3 * mB ** 2 * np.cos(2 * theta) * np.sin(
        theta) + 4 * LT ** 2 * R ** 2 * dtheta ** 2 * dx ** 2 * mB * mW * np.sin(
        2 * theta) - 4 * LT ** 2 * R ** 3 * TL * dtheta * dx * mB * np.cos(
        2 * theta) - 4 * LT ** 2 * R ** 3 * TR * dtheta * dx * mB * np.cos(
        2 * theta) + 2 * LT ** 4 * R ** 2 * dphi ** 2 * dtheta ** 2 * mB ** 2 * np.sin(2 * theta) * np.cos(
        theta) ** 2 - 16 * LT ** 4 * R ** 2 * dtheta ** 4 * dx ** 2 * mB ** 2 * np.cos(2 * theta) * np.sin(
        2 * theta) + 8 * LT ** 4 * R ** 2 * dtheta ** 4 * dx ** 2 * mB ** 2 * np.sin(2 * theta) * np.cos(
        theta) ** 2 + 8 * LT ** 3 * R ** 2 * dtheta ** 3 * dx * mB * mW * np.sin(
        theta) - 8 * LT ** 5 * R ** 2 * dtheta ** 5 * dx * mB ** 2 * np.cos(2 * theta) * np.sin(
        theta) - 24 * LT ** 5 * R ** 2 * dtheta ** 5 * dx * mB ** 2 * np.sin(2 * theta) * np.cos(
        theta) - 36 * LT ** 2 * R ** 2 * dtheta ** 2 * dx ** 4 * mB ** 2 * np.cos(theta) * np.sin(
        theta) + 16 * LT ** 5 * R ** 2 * dtheta ** 5 * dx * mB ** 2 * np.cos(theta) ** 2 * np.sin(
        theta) + 4 * LT ** 3 * R ** 2 * dtheta ** 2 * g * mB ** 2 * np.cos(theta) ** 2 * np.sin(
        theta) + 6 * LT ** 3 * R ** 2 * dphi ** 2 * dtheta * dx * mB ** 2 * np.sin(2 * theta) * np.cos(
        theta) + 12 * LT ** 2 * R ** 2 * dtheta * dx * g * mB ** 2 * np.cos(theta) * np.sin(theta)) / (2 * (
            IB * IW + IB * R ** 2 * mW + 6 * LT ** 6 * R ** 2 * dtheta ** 4 * mB ** 2 + 12 * LT ** 2 * R ** 2 * dx ** 4 * mB ** 2 + 6 * IW * LT ** 4 * dtheta ** 2 * mB + 4 * IW * LT ** 2 * dx ** 2 * mB + 3 * IB * R ** 2 * dx ** 2 * mB + 6 * LT ** 4 * R ** 2 * dtheta ** 2 * dx ** 2 * mB ** 2 + IB * LT ** 2 * R ** 2 * dtheta ** 2 * mB + 6 * LT ** 4 * R ** 2 * dtheta ** 2 * mB * mW + 4 * LT ** 2 * R ** 2 * dx ** 2 * mB * mW - 6 * LT ** 6 * R ** 2 * dtheta ** 4 * mB ** 2 * np.cos(
        theta) ** 2 + 6 * LT ** 2 * R ** 2 * dx ** 4 * mB ** 2 * np.cos(
        2 * theta) - 18 * LT ** 2 * R ** 2 * dx ** 4 * mB ** 2 * np.cos(
        theta) ** 2 + 2 * IW * LT ** 2 * dx ** 2 * mB * np.cos(
        2 * theta) - 6 * LT ** 4 * R ** 2 * dtheta ** 2 * dx ** 2 * mB ** 2 * np.cos(
        2 * theta) + 12 * LT ** 4 * R ** 2 * dtheta ** 2 * dx ** 2 * mB ** 2 * np.cos(
        theta) ** 2 + 12 * IW * LT ** 3 * dtheta * dx * mB * np.cos(
        theta) + 2 * IB * LT ** 2 * R ** 2 * dtheta ** 2 * mB * np.cos(
        theta) ** 2 + 24 * LT ** 3 * R ** 2 * dtheta * dx ** 3 * mB ** 2 * np.cos(
        theta) + 12 * LT ** 5 * R ** 2 * dtheta ** 3 * dx * mB ** 2 * np.cos(
        theta) + 2 * LT ** 2 * R ** 2 * dx ** 2 * mB * mW * np.cos(
        2 * theta) - 24 * LT ** 3 * R ** 2 * dtheta * dx ** 3 * mB ** 2 * np.cos(
        theta) ** 3 + 12 * LT ** 3 * R ** 2 * dtheta * dx * mB * mW * np.cos(
        theta) - 12 * LT ** 4 * R ** 2 * dtheta ** 2 * dx ** 2 * mB ** 2 * np.cos(2 * theta) * np.cos(
        theta) ** 2 + 6 * IB * LT * R ** 2 * dtheta * dx * mB * np.cos(
        theta) - 12 * LT ** 5 * R ** 2 * dtheta ** 3 * dx * mB ** 2 * np.cos(2 * theta) * np.cos(theta))))

    ddphi = (-(2 * R ** 2 * (
            R * TR - R * TL + 2 * LT ** 2 * b * dphi * dtheta * mB * np.cos(theta) * np.sin(theta))) / (b * (
            2 * IV * R ** 2 + IW * b ** 2 + R ** 2 * b ** 2 * mW + 2 * LT ** 2 * R ** 2 * mB * np.sin(theta) ** 2)))

    ddx = ((R ** 2 * (IB * R * TL + IB * R * TR + 12 * LT ** 7 * dtheta ** 6 * mB ** 2 * np.sin(
        theta) + 24 * LT ** 6 * dtheta ** 5 * dx * mB ** 2 * np.sin(
        2 * theta) + 24 * LT ** 3 * dtheta ** 2 * dx ** 4 * mB ** 2 * np.sin(
        theta) + 28 * LT ** 5 * dtheta ** 4 * dx ** 2 * mB ** 2 * np.sin(
        theta) + 2 * IB * LT ** 3 * dtheta ** 4 * mB * np.sin(
        theta) + 8 * LT ** 4 * dtheta ** 3 * dx ** 3 * mB ** 2 * np.sin(
        2 * theta) + 6 * LT ** 4 * R * TL * dtheta ** 2 * mB + 4 * LT ** 2 * R * TL * dx ** 2 * mB + 6 * LT ** 4 * R * TR * dtheta ** 2 * mB + 4 * LT ** 2 * R * TR * dx ** 2 * mB + 2 * LT ** 2 * R * TL * dx ** 2 * mB * np.cos(
        2 * theta) + 2 * LT ** 2 * R * TR * dx ** 2 * mB * np.cos(
        2 * theta) + 48 * LT ** 4 * dtheta ** 3 * dx ** 3 * mB ** 2 * np.cos(theta) * np.sin(
        theta) + 4 * IB * LT ** 2 * dtheta ** 3 * dx * mB * np.sin(
        2 * theta) - 3 * LT ** 5 * dphi ** 2 * dtheta ** 2 * mB ** 2 * np.sin(2 * theta) * np.cos(
        theta) - 3 * LT ** 3 * dphi ** 2 * dx ** 2 * mB ** 2 * np.sin(2 * theta) * np.cos(
        theta) + 12 * LT ** 3 * dtheta ** 2 * dx ** 4 * mB ** 2 * np.cos(2 * theta) * np.sin(
        theta) - 12 * LT ** 3 * dtheta ** 2 * dx ** 4 * mB ** 2 * np.sin(2 * theta) * np.cos(
        theta) + 4 * LT ** 5 * dtheta ** 4 * dx ** 2 * mB ** 2 * np.cos(2 * theta) * np.sin(
        theta) + 36 * LT ** 5 * dtheta ** 4 * dx ** 2 * mB ** 2 * np.sin(2 * theta) * np.cos(
        theta) - 32 * LT ** 5 * dtheta ** 4 * dx ** 2 * mB ** 2 * np.cos(theta) ** 2 * np.sin(
        theta) - 2 * LT ** 4 * dphi ** 2 * dtheta * dx * mB ** 2 * np.sin(
        2 * theta) - 6 * LT ** 4 * dtheta ** 2 * g * mB ** 2 * np.cos(theta) * np.sin(
        theta) - 6 * LT ** 2 * dx ** 2 * g * mB ** 2 * np.cos(theta) * np.sin(
        theta) + 6 * IB * LT * dtheta ** 2 * dx ** 2 * mB * np.sin(
        theta) + 8 * LT ** 4 * dtheta ** 3 * dx ** 3 * mB ** 2 * np.cos(2 * theta) * np.sin(
        2 * theta) - 16 * LT ** 4 * dtheta ** 3 * dx ** 3 * mB ** 2 * np.sin(2 * theta) * np.cos(
        theta) ** 2 - 4 * LT ** 3 * dtheta * dx * g * mB ** 2 * np.sin(
        theta) - 8 * LT ** 3 * dtheta * dx * g * mB ** 2 * np.cos(theta) ** 2 * np.sin(
        theta) + 12 * LT ** 3 * R * TL * dtheta * dx * mB * np.cos(
        theta) + 12 * LT ** 3 * R * TR * dtheta * dx * mB * np.cos(
        theta) - 4 * LT ** 4 * dphi ** 2 * dtheta * dx * mB ** 2 * np.sin(2 * theta) * np.cos(theta) ** 2)) / (2 * (
            IB * IW + IB * R ** 2 * mW + 6 * LT ** 6 * R ** 2 * dtheta ** 4 * mB ** 2 + 12 * LT ** 2 * R ** 2 * dx ** 4 * mB ** 2 + 6 * IW * LT ** 4 * dtheta ** 2 * mB + 4 * IW * LT ** 2 * dx ** 2 * mB + 3 * IB * R ** 2 * dx ** 2 * mB + 6 * LT ** 4 * R ** 2 * dtheta ** 2 * dx ** 2 * mB ** 2 + IB * LT ** 2 * R ** 2 * dtheta ** 2 * mB + 6 * LT ** 4 * R ** 2 * dtheta ** 2 * mB * mW + 4 * LT ** 2 * R ** 2 * dx ** 2 * mB * mW - 6 * LT ** 6 * R ** 2 * dtheta ** 4 * mB ** 2 * np.cos(
        theta) ** 2 + 6 * LT ** 2 * R ** 2 * dx ** 4 * mB ** 2 * np.cos(
        2 * theta) - 18 * LT ** 2 * R ** 2 * dx ** 4 * mB ** 2 * np.cos(
        theta) ** 2 + 2 * IW * LT ** 2 * dx ** 2 * mB * np.cos(
        2 * theta) - 6 * LT ** 4 * R ** 2 * dtheta ** 2 * dx ** 2 * mB ** 2 * np.cos(
        2 * theta) + 12 * LT ** 4 * R ** 2 * dtheta ** 2 * dx ** 2 * mB ** 2 * np.cos(
        theta) ** 2 + 12 * IW * LT ** 3 * dtheta * dx * mB * np.cos(
        theta) + 2 * IB * LT ** 2 * R ** 2 * dtheta ** 2 * mB * np.cos(
        theta) ** 2 + 24 * LT ** 3 * R ** 2 * dtheta * dx ** 3 * mB ** 2 * np.cos(
        theta) + 12 * LT ** 5 * R ** 2 * dtheta ** 3 * dx * mB ** 2 * np.cos(
        theta) + 2 * LT ** 2 * R ** 2 * dx ** 2 * mB * mW * np.cos(
        2 * theta) - 24 * LT ** 3 * R ** 2 * dtheta * dx ** 3 * mB ** 2 * np.cos(
        theta) ** 3 + 12 * LT ** 3 * R ** 2 * dtheta * dx * mB * mW * np.cos(
        theta) - 12 * LT ** 4 * R ** 2 * dtheta ** 2 * dx ** 2 * mB ** 2 * np.cos(2 * theta) * np.cos(
        theta) ** 2 + 6 * IB * LT * R ** 2 * dtheta * dx * mB * np.cos(
        theta) - 12 * LT ** 5 * R ** 2 * dtheta ** 3 * dx * mB ** 2 * np.cos(2 * theta) * np.cos(theta))))

    dpx = np.cos(phi) * dx
    dpy = np.sin(phi) * dx

    return [dtheta, ddtheta, dphi, ddphi, dx, ddx, dpx, dpy]


def create_input(label, id_name, val):
    return html.Div([
        html.Label(label, style={'fontWeight': 'bold', 'marginRight': '10px'}),
        dcc.Input(id=id_name, type='number', value=val, step=0.01, style={'width': '80px'})
    ], style={'padding': '5px'})


app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Robot balansujący"),

    html.Div([
        html.Div([
            html.H4("Nastawy PID"),
            create_input("Kp:", "input-kp", 200),
            create_input("Ti:", "input-ti", 0.5),
            create_input("Td:", "input-td", 0.015),
        ], style={'border': '1px solid #ddd', 'padding': '10px', 'marginRight': '10px'}),

        html.Div([
            html.H4("Właściwości robota"),
            create_input("Masa korpusu (mB):", "input-mB", 1),
            create_input("Masa koła (mW):", "input-mW", 0.5),
            create_input("Długość (LT):", "input-LT", 0.2),
            create_input("Promień (R):", "input-R", 0.05),
        ], style={'border': '1px solid #ddd', 'padding': '10px', 'marginRight': '10px'}),

        html.Div([
            html.H4("Parametry fizyczne"),
            create_input("Bezwładność korpusu (IB):", "input-IB", 0.1),
            create_input("Bezwładność koła (IW):", "input-IW", 0.02),
            create_input("Bezwładność obrotu (IV):", "input-IV", 0.05),
            create_input("Grawitacja (g):", "input-g", 9.81),
            create_input("Szerokość (b):", "input-b", 0.3),
        ], style={'border': '1px solid #ddd', 'padding': '10px', 'marginRight': '10px'}),

        html.Div([
            html.Button('Oblicz symulację', id='btn-run', n_clicks=0,
                        style={'height': '50px', 'width': '150px', 'backgroundColor': '#0074D9', 'color': 'white',
                               'fontSize': '16px', 'cursor': 'pointer'})
        ], style={'display': 'flex', 'alignItems': 'center'})

    ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '20px'}),

    dcc.Loading(
        id="loading-graph",
        type="default",
        children=dcc.Graph(id="sim-graph")
    )
])


@app.callback(
    Output("sim-graph", "figure"),
    Input("btn-run", "n_clicks"),
    State("input-kp", "value"),
    State("input-ki", "value"),
    State("input-kd", "value"),
    State("input-IB", "value"),
    State("input-IW", "value"),
    State("input-IV", "value"),
    State("input-R", "value"),
    State("input-b", "value"),
    State("input-LT", "value"),
    State("input-mB", "value"),
    State("input-mW", "value"),
    State("input-g", "value"),
)
def update_simulation(n_clicks, Kp, Ki, Kd, IB, IW, IV, R, b, LT, mB, mW, g):
    if n_clicks == 0:
        return go.Figure()

    pid_sim = PID(Kp, Ki, Kd, 1000)

    params = (IB, IV, R, b, LT, mB, mW, g, IW)

    sim_time = 5
    t = np.linspace(0, sim_time, 300)
    initial_state = [0.01, 0, 3.14 / 4, 0, 0, 0, 0, 0]

    solution = odeint(equations, initial_state, t, args=(params, pid_sim))

    theta = solution[:, 0]
    phi = solution[:, 2]
    x = solution[:, 4]
    px = solution[:, 6]
    py = solution[:, 7]

    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=["Trajektoria", "Theta (pochylenie)", "Phi", "Pozycja X"]
    )

    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="Trajectory"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="θ"), row=1, col=2)
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="φ"), row=1, col=3)
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="x"), row=1, col=4)

    frames = []

    step = 1
    for k in range(1, len(t) + 1, step):
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(x=px[:k], y=py[:k], mode="lines"),
                    go.Scatter(x=t[:k], y=theta[:k], mode="lines"),
                    go.Scatter(x=t[:k], y=phi[:k], mode="lines"),
                    go.Scatter(x=t[:k], y=x[:k], mode="lines"),
                ],
                traces=[0, 1, 2, 3],
                name=str(k)
            )
        )

    fig.frames = frames

    fig.update_layout(
        height=500,
        showlegend=False,
        updatemenus=[{
            "type": "buttons",
            "buttons": [{
                "label": "Start",
                "method": "animate",
                "args": [
                    None,
                    {
                        "frame": {"duration": 20, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 0}
                    }
                ]
            }]
        }]
    )

    fig.update_xaxes(title_text="X", row=1, col=1)
    fig.update_yaxes(title_text="Y", row=1, col=1)
    fig.update_xaxes(title_text="Czas (s)", row=1, col=2)
    fig.update_xaxes(title_text="Czas (s)", row=1, col=3)
    fig.update_xaxes(title_text="Czas (s)", row=1, col=4)

    return fig


if __name__ == "__main__":
    app.run(debug=True)