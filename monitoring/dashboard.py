"""Terminal dashboard for live trading monitoring using rich."""

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class Dashboard:
    """Rich-based terminal dashboard for the Kalshi BTC trading bot."""

    def __init__(self):
        self.console = Console()
        self.data: dict = {}

    def update(self, **kwargs) -> None:
        """Merge new state into the dashboard data store.

        Expected keys:
            btc_price, spread, active_market, time_remaining, phase,
            prediction, confidence, position, daily_pnl, win_rate,
            trade_count, risk_status, model_name, features_ok,
            drift_alerts
        """
        self.data.update(kwargs)

    def render(self) -> str:
        """Build a rich-formatted string of the current dashboard state."""
        d = self.data

        # --- header ---
        lines: list[str] = []
        lines.append("[bold cyan]Kalshi BTC 15m Trading Bot[/bold cyan]")
        lines.append("")

        # --- market ---
        ticker = d.get("active_market", "—")
        remaining = d.get("time_remaining", "—")
        phase = d.get("phase", "—")
        lines.append("[bold]Market[/bold]")
        lines.append(f"  Ticker:    {ticker}")
        lines.append(f"  Remaining: {remaining}")
        lines.append(f"  Phase:     {phase}")
        lines.append("")

        # --- price ---
        price = d.get("btc_price", "—")
        spread = d.get("spread", "—")
        if isinstance(price, (int, float)):
            price = f"${price:,.2f}"
        if isinstance(spread, (int, float)):
            spread = f"{spread}c"
        lines.append("[bold]Price[/bold]")
        lines.append(f"  BTC Mid:   {price}")
        lines.append(f"  Spread:    {spread}")
        lines.append("")

        # --- model ---
        prediction = d.get("prediction", "—")
        confidence = d.get("confidence", "—")
        model_name = d.get("model_name", "—")
        if prediction == "UP":
            prediction = "[green]UP[/green]"
        elif prediction == "DOWN":
            prediction = "[red]DOWN[/red]"
        elif prediction == "FLAT":
            prediction = "[yellow]FLAT[/yellow]"
        if isinstance(confidence, (int, float)):
            confidence = f"{confidence:.1f}%"
        lines.append("[bold]Model[/bold]")
        lines.append(f"  Prediction: {prediction}")
        lines.append(f"  Confidence: {confidence}")
        lines.append(f"  Model:      {model_name}")
        lines.append("")

        # --- position ---
        position = d.get("position", "Flat")
        if position and position != "Flat":
            position = f"[bold]{position}[/bold]"
        else:
            position = "[dim]Flat[/dim]"
        lines.append("[bold]Position[/bold]")
        lines.append(f"  {position}")
        lines.append("")

        # --- p&l ---
        daily_pnl = d.get("daily_pnl", 0)
        win_rate = d.get("win_rate", 0)
        trade_count = d.get("trade_count", 0)
        if isinstance(daily_pnl, (int, float)):
            pnl_dollars = daily_pnl / 100
            if pnl_dollars >= 0:
                pnl_str = f"[green]+${pnl_dollars:.2f}[/green]"
            else:
                pnl_str = f"[red]-${abs(pnl_dollars):.2f}[/red]"
        else:
            pnl_str = str(daily_pnl)
        if isinstance(win_rate, (int, float)):
            wr_str = f"{win_rate * 100:.1f}%"
        else:
            wr_str = str(win_rate)
        lines.append("[bold]P&L[/bold]")
        lines.append(f"  Daily:     {pnl_str}")
        lines.append(f"  Win Rate:  {wr_str}")
        lines.append(f"  Trades:    {trade_count}")
        lines.append("")

        # --- risk ---
        risk_status = d.get("risk_status", "OK")
        if risk_status == "OK":
            risk_str = "[green]OK[/green]"
        elif risk_status == "WARNING":
            risk_str = "[yellow]WARNING[/yellow]"
        else:
            risk_str = "[red]HALTED[/red]"
        lines.append("[bold]Risk[/bold]")
        lines.append(f"  Status: {risk_str}")
        lines.append("")

        # --- drift ---
        drift_alerts = d.get("drift_alerts", 0)
        features_ok = d.get("features_ok", True)
        if drift_alerts and drift_alerts > 0:
            drift_str = f"[yellow]{drift_alerts} feature(s) drifted[/yellow]"
        else:
            drift_str = "[green]No drift detected[/green]"
        lines.append("[bold]Drift[/bold]")
        lines.append(f"  {drift_str}")

        return "\n".join(lines)

    def display(self) -> None:
        """Print the rendered dashboard to the terminal."""
        rendered = self.render()
        panel = Panel(rendered, title="Trading Dashboard", border_style="blue")
        self.console.print(panel)
