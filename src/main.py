import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Load the pre-processed data
df = pd.read_csv("../notebooks/telecom_network_data_with_anomalies.csv")


# 1. Business impact analysis
def calculate_business_impact(df):
    """
    Estimate potential business impact of network anomalies
    """
    # Assumptions based on telecom industry standards
    COST_PER_MINUTE_DOWNTIME = 500  # Cost per minute of service downtime
    CUSTOMER_CHURN_COST = 5000  # USD per lost customer

    # Identify anomaly periods
    anomaly_periods = df[df["ensemble_anomaly"] == -1]

    # Estimate anomaly duration
    anomaly_groups = (
        anomaly_periods["timestamp"].diff() > pd.Timedelta(minutes=15)
    ).cumsum()
    downtime_events = anomaly_periods.groupby(anomaly_groups)

    # Calculate financial impact
    total_downtime_minutes = downtime_events.size().sum() * 15
    potential_revenue_loss = total_downtime_minutes * COST_PER_MINUTE_DOWNTIME
    potential_churn_impact = len(downtime_events) * CUSTOMER_CHURN_COST

    return {
        "total_downtime_minutes": total_downtime_minutes,
        "potential_revenue_lost": potential_revenue_loss,
        "potential_churn_impact": potential_churn_impact,
        "total_financial_impact": potential_revenue_loss + potential_churn_impact,
    }


# 2 Predictive maintenance recommendation


def generate_maintenance_recommendations(df, anomaly_analysis):
    """
    Generate actionable recommendations based on the anomaly analysis
    """
    recommendations = []

    # Traffic Load anomalies
    if (
        anomaly_analysis["traffic_load_analysis"]
        > anomaly_analysis["total_samples"] * 0.05
    ):
        recommendations.append(
            """
        1. Network Capacity Upgrade:
            - Increase bandwidth in high-traffic hours
            - Implement dynamic bandwidth allocation
            - Consider edge computing to distribute network load 
            """
        )

    # Latency Anomalies
    if anomaly_analysis["latency_anomalies"] > anomaly_analysis["total_samples"] * 0.03:
        recommendations.append(
            """
        2. Latency Optimization:
            - Audit and replace aging network infrastructure
            - Implement advanced traffic routing algorithm
            - Optimize network path selection
                               """
        )
    # Signal Strength anomalies
    if (
        anomaly_analysis["signal_strength_anomalies"]
        > anomaly_analysis["total_samples"] * 0.04
    ):
        recommendations.append(
            """
            3. Signal Coverage Improvement:
                - Conduct site survey for additional cell towers
                - Adjust antenna positioning
                - Implement Signal boosting technologies
                               
                               """
        )
    return recommendations


# 3. Executive summary dashboard
def create_executive_summary(df, business_impact, maintenance_recommendations):
    """
    Create Comprehensive executive summary

    Args:
        df (pandas dataframe): telecom data with anomaly
        business_impact (Dictionary like): business and churn financial impact
        maintenance_recommendations (List like): List of maintenance recommendations
    """
    summary = f"""
    # Network Performance executive summary
    
    ## Anomaly detection Insights
    - Total Network samples analyzed: {len(df)}
    - Anomalous Periods detected: {len(df[df['ensemble_anomaly']==-1])}
    - Anomaly Rate: {len(df[df['ensemble_anomaly']==-1]) / len(df) * 100:.2f}%
    
    ## Financial Impact Estimation
    - Potential Downtime: {business_impact['total_downtime_minutes']:.2f}minutes
    - Estimated Revenue Loss: ${business_impact['potential_revenue_lost']:.2f}
    - Potential Churn Cost: ${business_impact['potential_churn_impact']:.2f}
    - Total potential Financial Impact: ${business_impact['total_financial_impact']:.2f}
    
    ## Top Maintenance recommendations
    {char(10).join(maintenance_recommendations)}
    
    ## Key Performance Indicators
    - Average Traffic Load: {df['traffic_load_mbps'].mean():.2f} Mbps
    - Average Latency: {df['latency_ms'].mean():.2f} ms
    - Average Packet Loss: {df['packet_loss_percent'].mean():.2f} % 
    - Average Connection Success Rate: {df['Connection_success_rate'].mean():.2f}%
    """
    return summary


# 4. Visualization Functions
# ---------------------------
def create_interactive_anomaly_dashboard(df):
    """
    Create an Interactive dashboard for anomaly visualization

    Args:
        df
    """
    # Prepare data
    anomalies = df[df["ensemble_anomaly"] == -1]

    # Create subplot figure
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Traffic Load",
            "Latency",
            "Packet Loss",
            "Connection Success Rate",
        ),
    )
    # Traffic Load
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["traffic_load_mbps"],
            mode="markers",
            name="Anomalies",
            marker=dict(color="red", size=10),
        ),
        row=1,
        col=1,
    )

    # Latency
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["latency_ms"],
            mode="markers",
            name="Anomalies",
            marker=dict(color="red", size=10),
        ),
        row=2,
        col=1,
    )

    # Packet Loss
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["packet_loss_pct"],
            mode="markers",
            name="Anomalies",
            marker=dict(color="red", size=10),
        ),
        row=1,
        col=2,
    )

    # Connection Success Rate
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["conn_success_rate"],
            mode="markers",
            nmae="Anomalies",
            marker=dict(color="red", size=10),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=800, width=1200, title_text="Network Performance Anomaly Dashboard"
    )

    return fig


# Main Execution
# --------------


def main():
    # Anomaly Analysis
    anomaly_analysis = {
        "total_samples": len(df),
        "traffic_load_anomalies": len(
            df[
                (
                    (df["ensemble_anomaly"] == -1)
                    & (df["traffic_load_mbps"] > df["traffic_load_mbps"].quantile(0.95))
                )
            ]
        ),
        "latency_anomalies": len(
            df[
                (df["ensemble_anomaly"] == -1)
                & (df["letency_ms"] > df["latency_ms"].quantile(0.95))
            ]
        ),
        "singal_strength_anomalies": len(
            df[
                (df["ensemble_anomaly"] == -1)
                & (df["signal_strength_dbm"] > df["signal_strength_dbm"].quantile(0.05))
            ]
        ),
    }

    # Business Impact Calculation
    business_impact = calculate_business_impact(df)

    # Generate Maintenance recommendations
    maintenance_recommendations = generate_maintenance_recommendations(
        df, anomaly_analysis
    )

    # Create Executive Summary
    executive_summary = create_executive_summary(
        df,
        business_impact=business_impact,
        maintenance_recommendations=maintenance_recommendations,
    )

    # Save executive Summary
    with open("telecom_network_anomaly_report.md", "w") as f:
        f.write(executive_summary)

    # Create Interactive dashboard
    dashboard = create_interactive_anomaly_dashboard(df)
    dashboard.write_html("network_anomaly_dashboard.html")

    # Print Key Outputs
    print("Anomaly Detection Project Completed Successfully!")
    print(
        f"Total Potential Financial Impact: ${business_impact['total_financial_impact']:,.2f}"
    )
    print(f"Report saved to: telecom_network_anomaly_report.md")
    print(f"Interactive Dashboard saved to: network_anomaly_dashboard.html")


if __name__ == "__main__":
    main()
