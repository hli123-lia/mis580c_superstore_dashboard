import os
import re
import json
import threading
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html, dash_table, Input, Output, State
from dash.exceptions import PreventUpdate

# -------------------------------------------------------------------
# 1. Load & Preprocess Data
# -------------------------------------------------------------------
file_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(file_dir, "social_media_engagement_dataset.csv")
df = pd.read_csv(csv_path)

df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["Month"] = df["Timestamp"].dt.to_period("M").astype(str)
df["Interactions"] = df["Likes"] + df["Comments"] + df["Shares"] + df["Saves"]

# Sort orders & dropdown choices
platforms = sorted([p for p in df["Platform"].dropna().unique()])
categories = sorted([c for c in df["Category"].dropna().unique()])
tiers = ["Nano", "Micro", "Mid-tier", "Macro", "Mega"]
tiers = [t for t in tiers if t in df["Influencer_Tier"].unique()] or sorted(df["Influencer_Tier"].dropna().unique())
# Follower ranges per tier (from the dataset) shown next to tier names in the filter dropdown
TIER_RANGE_LABELS = {"Nano": "under 10K followers", "Micro": "10K\u201350K followers", "Mid-tier": "50K\u2013100K followers", "Macro": "100K+ followers"}
sentiments = ["Positive", "Neutral", "Negative"]
days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Instagram brand gradient palette
IG_YELLOW     = "#FCAF45"
IG_ORANGE     = "#F77737"
IG_RED        = "#FD1D1D"
IG_PINK       = "#E1306C"
IG_DARK_PINK  = "#C13584"
IG_PURPLE     = "#833AB4"
IG_BLUE       = "#5B51D8"
IG_ROYAL_BLUE = "#405DE6"

# Platform specific brand colors matching each platform's vector icon
PLATFORM_COLORS = {
    "Instagram": "#E1306C",
    "TikTok": "#000000",
    "YouTube": "#FF0000",
    "Facebook": "#1877F2",
    "Twitter": "#000000",
    "LinkedIn": "#0A66C2"
}

# Platform official vector logo data URIs (Base64 SVGs)
PLATFORM_LOGOS_B64 = {
    "Instagram": "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0OCA0OCIgd2lkdGg9IjQ4IiBoZWlnaHQ9IjQ4Ij48ZGVmcz48cmFkaWFsR3JhZGllbnQgaWQ9ImlnIiBjeD0iMTUlIiBjeT0iMTAwJSIgcj0iMTMwJSI+PHN0b3Agb2Zmc2V0PSIwJSIgc3RvcC1jb2xvcj0iI0ZGRDYwMCIvPjxzdG9wIG9mZnNldD0iMjUlIiBzdG9wLWNvbG9yPSIjRkYwMDY5Ii8+PHN0b3Agb2Zmc2V0PSI3MCUiIHN0b3AtY29sb3I9IiM3NjM4RkEiLz48c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9IiMzODAwODgiLz48L3JhZGlhbEdyYWRpZW50PjwvZGVmcz48cmVjdCB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIHJ4PSIxMiIgZmlsbD0idXJsKCNpZykiLz48cmVjdCB4PSI5IiB5PSI5IiB3aWR0aD0iMzAiIGhlaWdodD0iMzAiIHJ4PSI4IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMy41Ii8+PGNpcmNsZSBjeD0iMjQiIGN5PSIyNCIgcj0iNy41IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMy41Ii8+PGNpcmNsZSBjeD0iMzIiIGN5PSIxNiIgcj0iMi4yIiBmaWxsPSIjZmZmZmZmIi8+PC9zdmc+",
    "TikTok": "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjQ4IiBoZWlnaHQ9IjQ4Ij48cmVjdCB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHJ4PSI2IiBmaWxsPSIjMDAwMDAwIi8+PHBhdGggZmlsbD0iIzI1RjRFRSIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTAuNCwgLTAuNCkiIGQ9Ik0xOS41OSA2LjY5YTQuODMgNC44MyAwIDAgMS0zLjc3LTQuMjVWMmgtMy40NXYxMy42N2EyLjg5IDIuODkgMCAwIDEtMi44OCAyLjg5IDIuODkgMi44OSAwIDAgMS0yLjg5LTIuODkgMi44OSAyLjg5IDAgMCAxIDIuODktMi44OWMuMjggMCAuNTQuMDQuNzkuMVY5LjQyYTYuMzQgNi4zNCAwIDAgMC0uNzktLjA1QTYuMzMgNi4zMyAwIDAgMCAzIDE1LjdhNi4zMyA2LjMzIDAgMCAwIDYuMzMgNi4zMyA2LjMzIDYuMzMgMCAwIDAgNi4zMy02LjMzVjguODdhOC4yOCA4LjI4IDAgMCAwIDMuOTMuOTlWNi42OXoiLz48cGF0aCBmaWxsPSIjRkUyQzU1IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjQsIDAuNCkiIGQ9Ik0xOS41OSA2LjY5YTQuODMgNC44MyAwIDAgMS0zLjc3LTQuMjVWMmgtMy40NXYxMy42N2EyLjg5IDIuODkgMCAwIDEtMi44OCAyLjg5IDIuODkgMi44OSAwIDAgMS0yLjg5LTIuODkgMi44OSAyLjg5IDAgMCAxIDIuODktMi44OWMuMjggMCAuNTQuMDQuNzkuMVY5LjQyYTYuMzQgNi4zNCAwIDAgMC0uNzktLjA1QTYuMzMgNi4zMyAwIDAgMCAzIDE1LjdhNi4zMyA2LjMzIDAgMCAwIDYuMzMgNi4zMyA2LjMzIDYuMzMgMCAwIDAgNi4zMy02LjMzVjguODdhOC4yOCA4LjI4IDAgMCAwIDMuOTMuOTlWNi42OXoiLz48cGF0aCBmaWxsPSIjZmZmZmZmIiBkPSJNMTkuNTkgNi42OWE0LjgzIDQuODMgMCAwIDEtMy43Ny00LjI1VjJoLTMuNDV2MTMuNjdhMi44OSAyLjg5IDAgMCAxLTIuODggMi44OSAyLjg5IDIuODkgMCAwIDEtMi44OS0yLjg5IDIuODkgMi44OSAwIDAgMSAyLjg5LTIuODljLjI4IDAgLjU0LjA0Ljc5LjFWOS40MmE2LjM0IDYuMzQgMCAwIDAtLjc5LS4wNUE2LjMzIDYuMzMgMCAwIDAgMyAxNS43YTYuMzMgNi4zMyAwIDAgMCA2LjMzIDYuMzMgNi4zMyA2LjMzIDAgMCAwIDYuMzMtNi4zM1Y4Ljg3YTguMjggOC4yOCAwIDAgMCAzLjkzLjk5VjYuNjl6Ii8+PC9zdmc+",
    "YouTube": "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0OCA0OCIgd2lkdGg9IjQ4IiBoZWlnaHQ9IjQ4Ij48cmVjdCB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIHJ4PSIxMiIgZmlsbD0iI0ZGMDAwMCIvPjxwYXRoIGQ9Ik0yMCAxN2wxMiA3LTEyIDd6IiBmaWxsPSIjZmZmZmZmIi8+PC9zdmc+",
    "Facebook": "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0OCA0OCIgd2lkdGg9IjQ4IiBoZWlnaHQ9IjQ4Ij48cmVjdCB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIHJ4PSIxMiIgZmlsbD0iIzE4NzdGMiIvPjxwYXRoIGQ9Ik0yNyA0NFYyNmg2bDEtN2gtN3YtNC41YzAtMiAuNi0zLjUgMy41LTMuNUgzNFY0LjVDMzMuMyA0LjQgMzEgNCAyOC41IDQgMjMgNCAxOSA3LjQgMTkgMTR2NWgtNnY3aDZ2MThoOHoiIGZpbGw9IiNmZmZmZmYiLz48L3N2Zz4=",
    "Twitter": "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0OCA0OCIgd2lkdGg9IjQ4IiBoZWlnaHQ9IjQ4Ij48cmVjdCB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIHJ4PSIxMiIgZmlsbD0iIzAwMDAwMCIvPjxwYXRoIGQ9Ik0zMSAxMWg0LjVsLTkuOCAxMS4yTDM3IDM3aC05LjFsLTcuMS05LjNMMTIuNiAzN0g4LjFsMTAuNS0xMkw3LjUgMTFoOS4zbDYuNCA4LjVMMzEgMTF6bS0xLjYgMjMuM2gyLjVMMTguMSAxMy41aC0yLjdsMTQgMjAuOHoiIGZpbGw9IiNmZmZmZmYiLz48L3N2Zz4=",
    "LinkedIn": "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0OCA0OCIgd2lkdGg9IjQ4IiBoZWlnaHQ9IjQ4Ij48cmVjdCB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIHJ4PSIxMiIgZmlsbD0iIzBBNjZDMiIvPjxjaXJjbGUgY3g9IjE1IiBjeT0iMTUiIHI9IjMuNSIgZmlsbD0iI2ZmZmZmZiIvPjxyZWN0IHg9IjEyIiB5PSIyMSIgd2lkdGg9IjYiIGhlaWdodD0iMTciIHJ4PSIxIiBmaWxsPSIjZmZmZmZmIi8+PHBhdGggZD0iTTIyIDIxaDUuOHYyLjNoLjFjLjgtMS41IDIuNy0zIDUuNS0zIDUuOSAwIDcgMy45IDcgOC45VjM4aC02di03LjhjMC0xLjkgMC00LjMtMi42LTQuMy0yLjYgMC0zIDItMyA0LjJWMzhoLTYuOFYyMXoiIGZpbGw9IiNmZmZmZmYiLz48L3N2Zz4="
}

# Instagram gradient colorscale for Plotly
IG_COLORSCALE = [
    [0.0,  "#FCAF45"],
    [0.2,  "#F77737"],
    [0.4,  "#FD1D1D"],
    [0.6,  "#E1306C"],
    [0.75, "#C13584"],
    [0.88, "#833AB4"],
    [1.0,  "#405DE6"],
]

# TikTok Viral Spotlight presets. Counts read from TikTok on 2026-11-20 (rounded);
# engagement = (likes + comments + shares) / views.
TIKTOK_VIDEOS_METADATA = {
    "6768504823336815877": {
        "title": "Zach King - Flying Magic Broom",
        "views": "2.4B",
        "engagement": "1.3%",
        "shares": "1.1M",
        "format": "Magic Illusion"
    },
    "7332342275151760642": {
        "title": "Leah Halton - 'Summer' Lip Sync (Feb 2024)",
        "views": "1.3B",
        "engagement": "6.6%",
        "shares": "3M",
        "format": "Lip Sync / Trending Audio"
    },
    "7432101952424709422": {
        "title": "KATSEYE - 'you can sit with us at lunch' (Halloween 2024)",
        "views": "68.5M",
        "engagement": "16.4%",
        "shares": "495.7K",
        "format": "Dance / Group Choreography"
    }
}

# -------------------------------------------------------------------
# 2. Dash App Setup & Styles
# -------------------------------------------------------------------
app = dash.Dash(
    __name__,
    title="Social Media Intelligence Hub",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}]
)
server = app.server

app.index_string = '''<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            :root {
                /* Instagram Gradient Palette */
                --grad-1: #F58529; /* vibrant orange */
                --grad-2: #DD2A7B; /* pink / dark pink */
                --grad-3: #8134AF; /* purple */
                --grad-4: #515BD4; /* blue-violet */

                /* Content Card Glassmorphism */
                --card-bg: rgba(255, 255, 255, 0.82);
                --card-blur: 14px;
                --card-border: rgba(255, 255, 255, 0.60);
                --card-shadow: 0 8px 32px rgba(129, 52, 175, 0.12);

                /* Text Colors (WCAG AA Compliant) */
                --text-primary: #1f1f2e;
                --text-secondary: #6b6b80;
                --accent-purple: #8134AF;
                --accent-pink: #DD2A7B;

                /* Tab Gradient Underline */
                --tab-underline: linear-gradient(90deg, #F58529 0%, #DD2A7B 50%, #8134AF 100%);
            }

            * {
                box-sizing: border-box;
            }

            #chart-platform-content.pc-ready .js-plotly-plot { display: none !important; }
            .tab-parent { overflow: visible !important; }
            #chart-heatmap .nsewdrag { cursor: pointer !important; }
            #chart-content-type .barlayer path, #chart-category .barlayer path, #chart-tier .barlayer path { cursor: pointer !important; }

            body {
                margin: 0;
                padding: 0;
                min-height: 100vh;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                color: var(--text-primary);
                background-color: #fafafa;
                overflow-x: hidden;
            }

            /* Full page fixed background with 135deg gradient, pastel dissolution overlay, and organic blobs */
            .bg-fixed-gradient {
                position: fixed;
                inset: 0;
                z-index: -1;
                pointer-events: none;
                background:
                    /* Semi-transparent white overlay to soften lower page to pastel */
                    linear-gradient(to bottom, rgba(255, 255, 255, 0) 0%, rgba(255, 255, 255, 0.55) 38%, rgba(255, 255, 255, 0.85) 100%),
                    /* Organic blurred color blooms */
                    radial-gradient(circle at 12% 16%, rgba(245, 133, 41, 0.22) 0%, transparent 45%),
                    radial-gradient(circle at 88% 22%, rgba(221, 42, 123, 0.20) 0%, transparent 50%),
                    radial-gradient(circle at 48% 62%, rgba(129, 52, 175, 0.18) 0%, transparent 55%),
                    /* Base diagonal 135deg Instagram gradient */
                    linear-gradient(135deg, var(--grad-1) 0%, var(--grad-2) 35%, var(--grad-3) 70%, var(--grad-4) 100%);
                background-attachment: fixed;
            }

            .dashboard-container {
                min-height: 100vh;
                position: relative;
            }

            /* Header sits directly on the page gradient with no background or division line */
            .dashboard-header {
                position: relative;
                background: transparent;
                padding: 40px 48px 24px 48px;
                color: #ffffff;
                display: flex;
                flex-direction: column;
                align-items: center;
                text-align: center;
            }

            .header-center-content {
                display: flex;
                flex-direction: column;
                align-items: center;
                text-align: center;
                width: 100%;
                max-width: 1380px;
            }

            .header-action-btn-wrapper {
                position: absolute;
                top: 40px;
                right: 48px;
            }

            @media (max-width: 1440px) {
                .header-action-btn-wrapper {
                    position: static;
                    margin-top: 18px;
                }
            }

            .header-social-wrapper {
                position: absolute;
                top: 40px;
                left: 48px;
                z-index: 60;
                text-align: left;
            }

            @media (max-width: 1560px) {
                .header-social-wrapper {
                    position: relative;
                    top: auto;
                    left: auto;
                    margin-top: 36px;
                }
            }

            /* Tabs glass container */
            .tabs-glass-wrapper {
                background: rgba(255, 255, 255, 0.72);
                backdrop-filter: blur(var(--card-blur));
                -webkit-backdrop-filter: blur(var(--card-blur));
                border: 1px solid var(--card-border);
                border-radius: 18px;
                padding: 6px 14px;
                box-shadow: 0 6px 24px rgba(129, 52, 175, 0.08);
                margin-bottom: 24px;
            }

            .custom-tab {
                padding: 12px 24px !important;
                font-weight: 600 !important;
                font-size: 16px !important;
                color: var(--text-secondary) !important;
                border: none !important;
                background-color: transparent !important;
                transition: all 0.2s ease !important;
                cursor: pointer !important;
            }

            .custom-tab--selected {
                padding: 12px 24px !important;
                font-weight: 800 !important;
                font-size: 16px !important;
                color: var(--accent-purple) !important;
                border: none !important;
                background-color: transparent !important;
                position: relative !important;
            }

            .custom-tab--selected::after {
                content: "";
                position: absolute;
                bottom: 0;
                left: 12%;
                right: 12%;
                height: 3.5px;
                border-radius: 4px;
                background: var(--tab-underline);
            }

            #chart-heatmap .heatmaplayer image,
            #chart-platform-content .heatmaplayer image {
                opacity: 0 !important;
            }

            #chart-content-type g.point:nth-last-child(-n+3) path {
                fill: url(#contentTypeGradient) !important;
            }

            /* Tabs glass container */
            .tabs-glass-wrapper {
                background: rgba(255, 255, 255, 0.72);
                backdrop-filter: blur(var(--card-blur));
                -webkit-backdrop-filter: blur(var(--card-blur));
                border: 1px solid var(--card-border);
                border-radius: 18px;
                padding: 6px 14px;
                box-shadow: 0 6px 24px rgba(129, 52, 175, 0.08);
                margin-bottom: 24px;
            }

            .custom-tab {
                padding: 12px 24px !important;
                font-weight: 600 !important;
                font-size: 16px !important;
                color: var(--text-secondary) !important;
                border: none !important;
                background-color: transparent !important;
                transition: all 0.2s ease !important;
                cursor: pointer !important;
            }

            .custom-tab--selected {
                padding: 12px 24px !important;
                font-weight: 800 !important;
                font-size: 16px !important;
                color: var(--accent-purple) !important;
                border: none !important;
                background-color: transparent !important;
                position: relative !important;
            }

            .custom-tab--selected::after {
                content: "";
                position: absolute;
                bottom: 0;
                left: 12%;
                right: 12%;
                height: 3.5px;
                border-radius: 4px;
                background: var(--tab-underline);
            }

            #chart-heatmap .heatmaplayer image,
            #chart-platform-content .heatmaplayer image {
                opacity: 0 !important;
            }

            #chart-content-type g.point:nth-last-child(-n+3) path {
                fill: url(#contentTypeGradient) !important;
            }

            /* Tabs glass container */
            .tabs-glass-wrapper {
                background: rgba(255, 255, 255, 0.72);
                backdrop-filter: blur(var(--card-blur));
                -webkit-backdrop-filter: blur(var(--card-blur));
                border: 1px solid var(--card-border);
                border-radius: 18px;
                padding: 6px 14px;
                box-shadow: 0 6px 24px rgba(129, 52, 175, 0.08);
                margin-bottom: 24px;
            }

            .custom-tab {
                padding: 12px 24px !important;
                font-weight: 600 !important;
                font-size: 16px !important;
                color: var(--text-secondary) !important;
                border: none !important;
                background-color: transparent !important;
                transition: all 0.2s ease !important;
                cursor: pointer !important;
            }

            .custom-tab--selected {
                padding: 12px 24px !important;
                font-weight: 800 !important;
                font-size: 16px !important;
                color: var(--accent-purple) !important;
                border: none !important;
                background-color: transparent !important;
                position: relative !important;
            }

            .custom-tab--selected::after {
                content: "";
                position: absolute;
                bottom: 0;
                left: 12%;
                right: 12%;
                height: 3.5px;
                border-radius: 4px;
                background: var(--tab-underline);
            }

            #chart-heatmap .heatmaplayer image,
            #chart-platform-content .heatmaplayer image {
                opacity: 0 !important;
            }

            #chart-content-type g.point:nth-last-child(-n+3) path {
                fill: url(#contentTypeGradient) !important;
            }

            /* Tabs glass container */
            .tabs-glass-wrapper {
                background: rgba(255, 255, 255, 0.72);
                backdrop-filter: blur(var(--card-blur));
                -webkit-backdrop-filter: blur(var(--card-blur));
                border: 1px solid var(--card-border);
                border-radius: 18px;
                padding: 6px 14px;
                box-shadow: 0 6px 24px rgba(129, 52, 175, 0.08);
                margin-bottom: 24px;
            }

            .custom-tab {
                padding: 12px 24px !important;
                font-weight: 600 !important;
                font-size: 16px !important;
                color: var(--text-secondary) !important;
                border: none !important;
                background-color: transparent !important;
                transition: all 0.2s ease !important;
                cursor: pointer !important;
            }

            .custom-tab--selected {
                padding: 12px 24px !important;
                font-weight: 800 !important;
                font-size: 16px !important;
                color: var(--accent-purple) !important;
                border: none !important;
                background-color: transparent !important;
                position: relative !important;
            }

            .custom-tab--selected::after {
                content: "";
                position: absolute;
                bottom: 0;
                left: 12%;
                right: 12%;
                height: 3.5px;
                border-radius: 4px;
                background: var(--tab-underline);
            }

            #chart-heatmap .heatmaplayer image,
            #chart-platform-content .heatmaplayer image {
                opacity: 0 !important;
            }

            #chart-content-type g.point:nth-last-child(-n+3) path {
                fill: url(#contentTypeGradient) !important;
            }

            .like-burst-layer { position: fixed; left: 0; right: 0; bottom: 0; height: 0; z-index: 99999; pointer-events: none; }
            .like-burst { position: absolute; bottom: -60px; line-height: 1; opacity: 0; will-change: transform, opacity;
                animation: likeRise var(--dur, 2s) ease-out var(--delay, 0s) forwards; }
            @keyframes likeRise {
                0%   { transform: translate(0, 0) scale(0.4) rotate(-8deg); opacity: 0; }
                12%  { opacity: 1; }
                50%  { transform: translate(var(--sway, 20px), -45vh) scale(1) rotate(8deg); opacity: 1; }
                100% { transform: translate(calc(var(--sway, 20px) * -1), -85vh) scale(1.1) rotate(-6deg); opacity: 0; }
            }

            #comment-name.dash-input-element, #comment-name .dash-input-element { padding: 0 !important; }
            #comment-name input, #comment-text { box-sizing: border-box; }
            #comment-name input::placeholder, #comment-name::placeholder, #comment-text::placeholder {
                color: #9a9aa8 !important; -webkit-text-fill-color: #9a9aa8 !important; opacity: 1 !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
            <script>
                (function() {
                    var isUpdatingGradient = false;
                    function applyContentTypeGradient() {
                        if (isUpdatingGradient) return;
                        isUpdatingGradient = true;
                        try {
                            var chart = document.getElementById('chart-content-type');
                            if (!chart) return;
                            var svgs = chart.querySelectorAll('svg');
                            if (!svgs || svgs.length === 0) return;

                            svgs.forEach(function(svg) {
                                var defs = svg.querySelector('defs');
                                if (!defs) {
                                    defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
                                    svg.insertBefore(defs, svg.firstChild);
                                }
                                if (!defs.querySelector('#contentTypeGradient')) {
                                    var grad = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
                                    grad.setAttribute('id', 'contentTypeGradient');
                                    grad.setAttribute('x1', '0');
                                    grad.setAttribute('y1', '0');
                                    grad.setAttribute('x2', '1');
                                    grad.setAttribute('y2', '0');
                                    grad.setAttribute('gradientUnits', 'objectBoundingBox');
                                    var stops = [
                                        { offset: '0%', color: '#F58529' },
                                        { offset: '50%', color: '#DD2A7B' },
                                        { offset: '100%', color: '#8134AF' }
                                    ];
                                    stops.forEach(function(s) {
                                        var stop = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
                                        stop.setAttribute('offset', s.offset);
                                        stop.setAttribute('stop-color', s.color);
                                        grad.appendChild(stop);
                                    });
                                    defs.appendChild(grad);
                                }
                            });

                            var allPaths = Array.from(chart.querySelectorAll('.barlayer .point path, .bars .point path, g.point path, .barlayer path, path.point'));
                            var paths = allPaths.filter(function(p, idx, self) {
                                return self.indexOf(p) === idx && p.tagName && p.tagName.toLowerCase() === 'path' && p.getAttribute('d') && !p.classList.contains('yerror') && !p.classList.contains('xerror');
                            });

                            if (paths && paths.length >= 3) {
                                var sortedByTop = paths.slice().sort(function(a, b) {
                                    return a.getBoundingClientRect().top - b.getBoundingClientRect().top;
                                });
                                var top3 = new Set(sortedByTop.slice(0, 3));
                                for (var k = paths.length - 3; k < paths.length; k++) {
                                    top3.add(paths[k]);
                                }

                                paths.forEach(function(p) {
                                    if (top3.has(p)) {
                                        p.style.setProperty('fill', 'url(#contentTypeGradient)', 'important');
                                        p.setAttribute('fill', 'url(#contentTypeGradient)');
                                    } else {
                                        p.style.setProperty('fill', '#CFC8DC', 'important');
                                        p.setAttribute('fill', '#CFC8DC');
                                    }
                                });
                            }
                        } catch (err) {
                            console.error('applyContentTypeGradient error:', err);
                        } finally {
                            isUpdatingGradient = false;
                        }
                    }

                    var isUpdatingHeatmap = false;
                    function applyHeatmapRoundedCells() {
                        if (isUpdatingHeatmap) return;
                        isUpdatingHeatmap = true;
                        try {
                            var chart = document.getElementById('chart-heatmap');
                            if (!chart) return;
                            var gd = chart.querySelector('.js-plotly-plot') || chart;
                            if (!gd) return;

                            var imgs = chart.querySelectorAll('.heatmaplayer image, image.heatmap');
                            imgs.forEach(function(img) {
                                img.style.setProperty('opacity', '0', 'important');
                                img.setAttribute('opacity', '0');
                            });

                            var xa = gd._fullLayout && gd._fullLayout.xaxis;
                            var ya = gd._fullLayout && gd._fullLayout.yaxis;
                            var img = imgs.length ? imgs[0] : null;
                            var plotW = img ? parseFloat(img.getAttribute('width') || 0) : (xa ? xa._length : 0);
                            var plotH = img ? parseFloat(img.getAttribute('height') || 0) : (ya ? ya._length : 0);
                            var plotX = img ? parseFloat(img.getAttribute('x') || 0) : 0;
                            var plotY = img ? parseFloat(img.getAttribute('y') || 0) : 0;
                            if (!plotW || !plotH) return;

                            var zData = (gd.data && gd.data[0] && gd.data[0].z) || (gd._fullData && gd._fullData[0] && gd._fullData[0].z);
                            if (!zData || !zData.length) return;

                            var fd = (gd._fullData && gd._fullData[0]) || (gd.data && gd.data[0]) || {};
                            var zmin = (typeof fd.zmin === 'number') ? fd.zmin : 0;
                            var zmax = (typeof fd.zmax === 'number') ? fd.zmax : 40;
                            if (!(zmax > zmin)) { zmax = zmin + 1; }

                            var oldGroup = chart.querySelector('.custom-heatmap-cells');
                            if (oldGroup) oldGroup.remove();

                            var cellGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                            cellGroup.setAttribute('class', 'custom-heatmap-cells');
                            cellGroup.setAttribute('style', 'pointer-events: none;');

                            if (img && img.parentNode) {
                                img.parentNode.insertBefore(cellGroup, img);
                            } else {
                                var hlayer = chart.querySelector('.heatmaplayer') || chart.querySelector('svg.main-svg');
                                if (hlayer) hlayer.appendChild(cellGroup);
                            }

                            var STOPS = [
                                { t: 0.0, r: 0xFD, g: 0xE7, b: 0xCF },
                                { t: 0.2, r: 0xFB, g: 0xC9, b: 0x8E },
                                { t: 0.4, r: 0xF5, g: 0x85, b: 0x29 },
                                { t: 0.6, r: 0xE8, g: 0x50, b: 0x3F },
                                { t: 0.8, r: 0xDD, g: 0x2A, b: 0x7B },
                                { t: 1.0, r: 0x81, g: 0x34, b: 0xAF }
                            ];

                            function getHeatmapColor(val) {
                                if (val === null || val === undefined || isNaN(val)) return 'rgba(0,0,0,0)';
                                var norm = Math.max(0, Math.min(1, (Number(val) - zmin) / (zmax - zmin)));
                                for (var i = 0; i < STOPS.length - 1; i++) {
                                    var s0 = STOPS[i];
                                    var s1 = STOPS[i + 1];
                                    if (norm >= s0.t && norm <= s1.t) {
                                        var frac = (norm - s0.t) / (s1.t - s0.t);
                                        var r = Math.round(s0.r + frac * (s1.r - s0.r));
                                        var g = Math.round(s0.g + frac * (s1.g - s0.g));
                                        var b = Math.round(s0.b + frac * (s1.b - s0.b));
                                        return 'rgb(' + r + ',' + g + ',' + b + ')';
                                    }
                                }
                                return '#8134AF';
                            }

                            var numRows = zData.length;
                            var numCols = zData[0].length;
                            var gap = 2;
                            var colStep = plotW / numCols;
                            var rowStep = plotH / numRows;
                            var cellW = Math.max(1, colStep - gap);
                            var cellH = Math.max(1, rowStep - gap);

                            var html = '';
                            for (var r = 0; r < numRows; r++) {
                                for (var c = 0; c < numCols; c++) {
                                    var cellX = (plotX + c * colStep + (gap / 2)).toFixed(2);
                                    var cellY = (plotY + r * rowStep + (gap / 2)).toFixed(2);
                                    var val = zData[r][c];
                                    var color = getHeatmapColor(val);
                                    var normV = Math.max(0, Math.min(1, ((Number(val) || 0) - zmin) / (zmax - zmin)));
                                    var textColor = normV >= 0.65 ? 'rgba(255,255,255,0.85)' : 'rgba(31,31,46,0.55)';
                                    var hourText = (r === 0 ? "12 AM" : (r < 12 ? r + " AM" : (r === 12 ? "12 PM" : (r - 12) + " PM")));
                                    var midX = (parseFloat(cellX) + cellW / 2).toFixed(2);
                                    var midY = (parseFloat(cellY) + cellH / 2).toFixed(2);

                                    html += '<rect x="' + cellX + '" y="' + cellY + '" width="' + cellW.toFixed(2) + '" height="' + cellH.toFixed(2) + '" rx="4" ry="4" fill="' + color + '"/>';
                                    html += '<text x="' + midX + '" y="' + midY + '" text-anchor="middle" dominant-baseline="central" fill="' + textColor + '" font-size="9px" font-weight="400">' + hourText + '</text>';
                                }
                            }
                            var selMeta = gd.layout && gd.layout.meta && gd.layout.meta.sel;
                            if (selMeta && selMeta.length === 2) {
                                var sx = (plotX + selMeta[0] * colStep + gap / 2).toFixed(2);
                                var sy = (plotY + selMeta[1] * rowStep + gap / 2).toFixed(2);
                                html += '<rect x="' + sx + '" y="' + sy + '" width="' + cellW.toFixed(2) + '" height="' + cellH.toFixed(2) + '" rx="4" ry="4" fill="none" stroke="#1f1f2e" stroke-width="2.5"/>';
                            }
                            cellGroup.innerHTML = html;
                        } catch (err) {
                            console.error('applyHeatmapRoundedCells error:', err);
                        } finally {
                            isUpdatingHeatmap = false;
                        }
                    }

                    var isUpdatingPlatContent = false;
                    function applyPlatformContentCells() {
                        if (isUpdatingPlatContent) return;
                        isUpdatingPlatContent = true;
                        try {
                            var chart = document.getElementById('chart-platform-content');
                            if (!chart) return;
                            var gd = chart.querySelector('.js-plotly-plot');
                            var oldWrap = chart.querySelector('.pc-wrap');
                            var tr = gd && gd.data && gd.data[0];
                            if (!tr || !tr.z || !tr.z.length || !tr.x || !tr.y) {
                                if (oldWrap) oldWrap.remove();
                                chart.classList.remove('pc-ready');
                                chart._pcKey = null;
                                return;
                            }
                            var zData = tr.z, xs = tr.x, ys = tr.y;
                            var titleText = (gd.layout && gd.layout.title && gd.layout.title.text) || '';
                            var key = JSON.stringify([zData, xs, ys, titleText, Math.round(chart.clientWidth || 0)]);
                            if (oldWrap && chart._pcKey === key) return;
                            chart._pcKey = key;
                            if (oldWrap) oldWrap.remove();

                            var NS = 'http://www.w3.org/2000/svg';
                            var COLORS = ['#FDE7CF', '#FBC98E', '#F58529', '#E8503F', '#DD2A7B', '#8134AF'];
                            var CAP = (tr.zmax && isFinite(tr.zmax)) ? Number(tr.zmax) : 40.0;
                            var isRate = CAP <= 100;
                            function compact(v) { v = Number(v); if (v >= 1e9) return (v / 1e9).toFixed(1) + 'B'; if (v >= 1e6) return (v / 1e6).toFixed(1) + 'M'; if (v >= 1e3) return (v / 1e3).toFixed(0) + 'K'; return v.toFixed(0); }
                            var STEP = CAP / COLORS.length;
                            function stepIndex(v) { return Math.max(0, Math.min(COLORS.length - 1, Math.floor(Number(v) / STEP))); }

                            var rows = ys.length, cols = xs.length;
                            var availW = Math.max(300, chart.clientWidth || 0);
                            var leftMargin = 100, topMargin = 92, cellH = 42, gap = 3;
                            var cellW = Math.max(44, Math.floor((availW - leftMargin) / cols) - gap);
                            var gridW = cols * (cellW + gap), gridH = rows * (cellH + gap);
                            var W = leftMargin + gridW;
                            var H = topMargin + gridH + 90;

                            var wrap = document.createElement('div');
                            wrap.className = 'pc-wrap';
                            wrap.style.cssText = 'position:relative;width:100%;padding-bottom:20px;';
                            var svg = document.createElementNS(NS, 'svg');
                            svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
                            svg.setAttribute('preserveAspectRatio', 'xMinYMin meet');
                            svg.style.cssText = 'width:' + W + 'px;max-width:100%;height:auto;overflow:visible;display:block;font-family:Open Sans,verdana,arial,sans-serif;';
                            var tip = document.createElement('div');
                            tip.style.cssText = 'position:absolute;display:none;pointer-events:none;background:#fff;border:1px solid #8134AF;color:#1f1f2e;border-radius:6px;padding:4px 8px;font-size:13px;font-family:Open Sans,verdana,arial,sans-serif;white-space:nowrap;z-index:5;box-shadow:0 4px 12px rgba(0,0,0,0.12);';

                            function mk(tag, attrs, text) {
                                var el = document.createElementNS(NS, tag);
                                for (var k in attrs) el.setAttribute(k, attrs[k]);
                                if (text !== undefined) el.textContent = text;
                                svg.appendChild(el);
                                return el;
                            }

                            var m = titleText.match(/^<b>(.*?)<[/]b>(.*)$/);
                            var bold = m ? m[1] : titleText.replace(/<[^>]+>/g, '');
                            var rest = m ? m[2] : '';
                            var t = mk('text', { x: 33, y: 22, fill: '#8134AF', 'font-size': 17 });
                            var tb = document.createElementNS(NS, 'tspan'); tb.setAttribute('font-weight', '700'); tb.textContent = bold; t.appendChild(tb);
                            var tn = document.createElementNS(NS, 'tspan'); tn.textContent = rest; t.appendChild(tn);

                            mk('text', { x: leftMargin + gridW / 2, y: 56, 'text-anchor': 'middle', fill: '#1f1f2e', 'font-size': 13, 'font-weight': '700' }, 'Content Type');
                            var py = topMargin + gridH / 2;
                            mk('text', { x: 14, y: py, 'text-anchor': 'middle', fill: '#1f1f2e', 'font-size': 13, 'font-weight': '700', transform: 'rotate(-90 14 ' + py + ')' }, 'Platform');

                            for (var j = 0; j < cols; j++) {
                                var cx = leftMargin + j * (cellW + gap);
                                var parts = String(xs[j]).split(' ');
                                var lab = mk('text', { x: cx + cellW / 2, y: topMargin - 8 - (parts.length - 1) * 12, 'text-anchor': 'middle', fill: '#1f1f2e', 'font-size': 12 });
                                for (var q = 0; q < parts.length; q++) {
                                    var ts = document.createElementNS(NS, 'tspan');
                                    ts.setAttribute('x', cx + cellW / 2);
                                    if (q > 0) ts.setAttribute('dy', 12);
                                    ts.textContent = parts[q];
                                    lab.appendChild(ts);
                                }
                            }
                            for (var i = 0; i < rows; i++) {
                                var ry = topMargin + i * (cellH + gap);
                                mk('text', { x: leftMargin - 10, y: ry + cellH / 2, 'text-anchor': 'end', 'dominant-baseline': 'central', fill: '#1f1f2e', 'font-size': 12 }, ys[i]);
                                for (var j2 = 0; j2 < cols; j2++) {
                                    var x0 = leftMargin + j2 * (cellW + gap);
                                    var v = zData[i][j2];
                                    var empty = (v === null || v === undefined || isNaN(v));
                                    if (empty) {
                                        mk('rect', { x: x0, y: ry, width: cellW, height: cellH, rx: 6, ry: 6, fill: 'none', stroke: '#CFC8DC', 'stroke-width': 1.5, 'stroke-dasharray': '4 3' });
                                    } else {
                                        var idx = stepIndex(v);
                                        var rect = mk('rect', { x: x0, y: ry, width: cellW, height: cellH, rx: 6, ry: 6, fill: COLORS[idx] });
                                        mk('text', { x: x0 + cellW / 2, y: ry + cellH / 2, 'text-anchor': 'middle', 'dominant-baseline': 'central', fill: Number(v) >= CAP * 0.625 ? '#ffffff' : '#1f1f2e', 'font-size': 12, 'font-weight': '700', 'pointer-events': 'none' }, isRate ? Number(v).toFixed(1) : compact(v));
                                        (function(r, label) {
                                            r.addEventListener('mousemove', function(ev) {
                                                var box = wrap.getBoundingClientRect();
                                                tip.textContent = label;
                                                tip.style.display = 'block';
                                                tip.style.left = (ev.clientX - box.left + 12) + 'px';
                                                tip.style.top = (ev.clientY - box.top + 12) + 'px';
                                            });
                                            r.addEventListener('mouseleave', function() { tip.style.display = 'none'; });
                                        })(rect, ys[i] + ' · ' + xs[j2] + ' · ' + (isRate ? Number(v).toFixed(1) + '%' : Math.round(Number(v)).toLocaleString()));
                                    }
                                }
                            }

                            var legY = topMargin + gridH + 22, sw = gridW / COLORS.length;
                            for (var c = 0; c < COLORS.length; c++) {
                                mk('rect', { x: leftMargin + c * sw, y: legY, width: sw, height: 8, fill: COLORS[c] });
                            }
                            mk('text', { x: leftMargin, y: legY + 24, fill: '#6b6b80', 'font-size': 11 }, 'low');
                            mk('text', { x: leftMargin + COLORS.length * sw, y: legY + 24, 'text-anchor': 'end', fill: '#6b6b80', 'font-size': 11 }, isRate ? '40%+' : compact(CAP) + '+');

                            wrap.appendChild(svg);
                            wrap.appendChild(tip);
                            chart.appendChild(wrap);
                            chart.classList.add('pc-ready');
                        } catch (err) {
                            console.error('applyPlatformContentCells error:', err);
                        } finally {
                            isUpdatingPlatContent = false;
                        }
                    }

                    function attachChartHandlers() {
                        var chart = document.getElementById('chart-content-type');
                        if (chart) {
                            var plotDiv = chart.querySelector('.js-plotly-plot') || chart;
                            if (plotDiv && !plotDiv._hasGradientHandler) {
                                if (typeof plotDiv.on === 'function') {
                                    plotDiv.on('plotly_afterplot', applyContentTypeGradient);
                                    plotDiv._hasGradientHandler = true;
                                }
                                plotDiv.addEventListener('plotly_afterplot', applyContentTypeGradient);
                            }
                            applyContentTypeGradient();
                        }

                        var heatGraph = document.getElementById('chart-heatmap');
                        if (heatGraph) {
                            var heatPlotDiv = heatGraph.querySelector('.js-plotly-plot') || heatGraph;
                            if (heatPlotDiv && !heatPlotDiv._hasHeatmapHandler) {
                                if (typeof heatPlotDiv.on === 'function') {
                                    heatPlotDiv.on('plotly_afterplot', applyHeatmapRoundedCells);
                                    heatPlotDiv._hasHeatmapHandler = true;
                                }
                                heatPlotDiv.addEventListener('plotly_afterplot', applyHeatmapRoundedCells);
                            }
                            applyHeatmapRoundedCells();
                        }

                        var platGraph = document.getElementById('chart-platform-content');
                        if (platGraph) {
                            var platPlotDiv = platGraph.querySelector('.js-plotly-plot') || platGraph;
                            if (platPlotDiv && !platPlotDiv._hasPlatContentHandler) {
                                if (typeof platPlotDiv.on === 'function') {
                                    platPlotDiv.on('plotly_afterplot', applyPlatformContentCells);
                                    platPlotDiv._hasPlatContentHandler = true;
                                }
                                platPlotDiv.addEventListener('plotly_afterplot', applyPlatformContentCells);
                            }
                            applyPlatformContentCells();
                        }
                    }

                    window.addEventListener('load', attachChartHandlers);
                    document.addEventListener('DOMContentLoaded', attachChartHandlers);
                    window.addEventListener('resize', function() {
                        applyContentTypeGradient();
                        applyHeatmapRoundedCells();
                        applyPlatformContentCells();
                    });
                    document.addEventListener('click', function() {
                        setTimeout(attachChartHandlers, 100);
                        setTimeout(attachChartHandlers, 400);
                    });

                    document.addEventListener('click', function(ev) {
                        var btn = ev.target && ev.target.closest ? ev.target.closest('#like-btn') : null;
                        if (!btn) return;
                        if ((btn.textContent || '').indexOf('Liked') !== -1) return; // un-liking: no animation
                        var layer = document.querySelector('.like-burst-layer');
                        if (!layer) {
                            layer = document.createElement('div');
                            layer.className = 'like-burst-layer';
                            document.body.appendChild(layer);
                        }
                        var n = 16;
                        for (var i = 0; i < n; i++) {
                            var el = document.createElement('span');
                            el.className = 'like-burst';
                            el.textContent = '👍';
                            el.style.left = (8 + Math.random() * 84) + '%';
                            el.style.fontSize = (28 + Math.random() * 30) + 'px';
                            el.style.setProperty('--dur', (1.8 + Math.random() * 1.2).toFixed(2) + 's');
                            el.style.setProperty('--delay', (Math.random() * 0.5).toFixed(2) + 's');
                            el.style.setProperty('--sway', (Math.random() * 60 - 30).toFixed(0) + 'px');
                            layer.appendChild(el);
                            (function(e) { setTimeout(function() { if (e.parentNode) e.parentNode.removeChild(e); }, 3800); })(el);
                        }
                    });
                    setInterval(attachChartHandlers, 1000);
                })();
            </script>
        </footer>
    </body>
</html>'''

# Reusable Frosted-Glass CSS Styles (WCAG AA Compliant)
CARD_BOX = {
    "backgroundColor": "rgba(255, 255, 255, 0.82)",
    "borderRadius": "20px",
    "padding": "22px 24px",
    "boxShadow": "0 8px 32px rgba(129, 52, 175, 0.12)",
    "border": "1px solid rgba(255, 255, 255, 0.60)",
}

BAR_BANNER_STYLE = {
    "backgroundColor": "rgba(129, 52, 175, 0.12)", "color": "#8134AF",
    "border": "1px solid rgba(129, 52, 175, 0.40)", "borderRadius": "999px",
    "padding": "5px 14px", "fontSize": "12.5px", "fontWeight": "700",
}

HEAT_HINT_STYLE = {
    "backgroundColor": "rgba(129, 52, 175, 0.12)", "color": "#8134AF",
    "border": "1px solid rgba(129, 52, 175, 0.40)", "borderRadius": "999px",
    "padding": "4px 12px", "fontSize": "12px", "fontWeight": "700", "pointerEvents": "none",
}

TAB_STYLE = {
    "padding": "12px 24px",
    "fontWeight": "600",
    "fontSize": "16px",
    "color": "#6b6b80",
    "border": "none",
    "backgroundColor": "transparent"
}

TAB_SELECTED_STYLE = {
    "padding": "12px 24px",
    "fontWeight": "800",
    "fontSize": "16px",
    "color": "#8134AF",
    "border": "none",
    "backgroundColor": "transparent"
}

# -------------------------------------------------------------------
# Social widget (like / subscribe / comments) - totals and comments are saved in a small
# JSON file next to this app, so everyone using the same running app sees the same numbers.
# Delete engagement_data.json to reset everything.
# -------------------------------------------------------------------
ENGAGEMENT_FILE = os.path.join(file_dir, "engagement_data.json")
_engagement_lock = threading.Lock()


def _load_engagement():
    try:
        with open(ENGAGEMENT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "likes": int(data.get("likes", 0)),
            "subscribers": int(data.get("subscribers", 0)),
            "comments": list(data.get("comments", [])),
        }
    except Exception:
        return {"likes": 0, "subscribers": 0, "comments": []}


engagement_state = _load_engagement()


def _save_engagement():
    try:
        tmp_path = ENGAGEMENT_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(engagement_state, f)
        os.replace(tmp_path, ENGAGEMENT_FILE)
    except Exception as err:
        print(f"Could not save engagement data: {err}")


_BLOCKED_WORDS = re.compile(r"\b\w*(?:fuck|shit|bitch|asshole)\w*\b", re.IGNORECASE)


def _clean_text(value, max_len):
    value = re.sub(r"\s+", " ", str(value or "")).strip()[:max_len]
    return _BLOCKED_WORDS.sub(lambda m: "*" * len(m.group()), value)


def compact_count(n):
    """1,234 -> 1.2K, 2,300,000 -> 2.3M (numbers under 1,000 stay as they are)."""
    n = int(n)
    if n < 1000:
        return str(n)
    for suffix, size in (("K", 1e3), ("M", 1e6), ("B", 1e9)):
        v = n / size
        if round(v, 1) < 1000 or suffix == "B":
            return f"{v:.1f}".rstrip("0").rstrip(".") + suffix


def social_btn_style(active=False):
    style = {
        "color": "#ffffff",
        "borderRadius": "12px",
        "padding": "8px 12px",
        "fontSize": "13px",
        "fontWeight": "700",
        "cursor": "pointer",
        "boxShadow": "0 4px 16px rgba(0, 0, 0, 0.12)",
        "textShadow": "0 1px 10px rgba(0, 0, 0, 0.25)",
        "transition": "all 0.2s ease",
        "whiteSpace": "nowrap",
    }
    if active:
        style.update({
            "background": "linear-gradient(135deg, #F58529 0%, #DD2A7B 50%, #8134AF 100%)",
            "border": "1px solid rgba(255, 255, 255, 0.70)",
        })
    else:
        style.update({
            "backgroundColor": "rgba(255, 255, 255, 0.25)",
            "border": "1px solid rgba(255, 255, 255, 0.50)",
            "backdropFilter": "blur(14px)",
            "-webkitBackdropFilter": "blur(14px)",
        })
    return style


COMMENT_PANEL_STYLE = {
    "display": "none",
    "position": "absolute",
    "top": "100%",
    "left": "0",
    "marginTop": "10px",
    "width": "360px",
    "maxWidth": "90vw",
    "backgroundColor": "rgba(255, 255, 255, 0.97)",
    "borderRadius": "16px",
    "padding": "16px",
    "boxShadow": "0 12px 40px rgba(129, 52, 175, 0.25)",
    "border": "1px solid rgba(129, 52, 175, 0.15)",
    "zIndex": 100,
    "textAlign": "left",
}

_COMMENT_INPUT_STYLE = {
    "width": "100%",
    "boxSizing": "border-box",
    "padding": "9px 12px",
    "fontSize": "14px",
    "borderRadius": "10px",
    "border": "1px solid rgba(0, 0, 0, 0.15)",
    "marginBottom": "8px",
    "fontFamily": "inherit",
    "color": "#000000",
    "WebkitTextFillColor": "#000000",
    "backgroundColor": "#ffffff",
    "textShadow": "none",
}


def render_comments(comments):
    if not comments:
        return html.Div("No comments yet - be the first!", style={"color": "#6b6b80", "fontSize": "14px", "padding": "8px 0"})
    return [
        html.Div(
            [
                html.Div([
                    html.Span(c.get("name", "Anonymous"), style={"fontWeight": "800", "color": "#8134AF", "fontSize": "14px"}),
                    html.Span(c.get("time", ""), style={"color": "#9a9aae", "fontSize": "12px", "marginLeft": "8px"}),
                ]),
                html.Div(c.get("text", ""), style={"color": "#1f1f2e", "fontSize": "14px", "lineHeight": "1.4", "marginTop": "2px", "wordBreak": "break-word"}),
            ],
            style={"padding": "8px 0", "borderBottom": "1px solid rgba(0, 0, 0, 0.06)"},
        )
        for c in comments[:30]
    ]


def apply_social_action(trigger, liked, subscribed, name, text):
    """Applies one click (like, subscribe or post) and returns the new personal flags plus the shared totals."""
    liked, subscribed, clear_text = bool(liked), bool(subscribed), False
    with _engagement_lock:
        if trigger == "like-btn":
            liked = not liked
            engagement_state["likes"] = max(0, engagement_state["likes"] + (1 if liked else -1))
            _save_engagement()
        elif trigger == "sub-btn":
            subscribed = not subscribed
            engagement_state["subscribers"] = max(0, engagement_state["subscribers"] + (1 if subscribed else -1))
            _save_engagement()
        elif trigger == "post-comment-btn":
            clean = _clean_text(text, 200)
            if clean:
                engagement_state["comments"].insert(0, {
                    "name": _clean_text(name, 24) or "Anonymous",
                    "text": clean,
                    "time": datetime.now().strftime("%b %d, %I:%M %p"),
                })
                del engagement_state["comments"][200:]
                _save_engagement()
                clear_text = True
        snapshot = {
            "likes": engagement_state["likes"],
            "subscribers": engagement_state["subscribers"],
            "comments": list(engagement_state["comments"]),
        }
    return liked, subscribed, clear_text, snapshot



# -------------------------------------------------------------------
# 3. Application Layout
# -------------------------------------------------------------------
app.layout = html.Div(
    className="dashboard-container",
    children=[
        # Full Page Fixed Gradient Background Layer
        html.Div(className="bg-fixed-gradient"),

        # Top Header Banner (Sits directly on page gradient - Centered)
        html.Div(
            className="dashboard-header",
            children=[
                html.Div(
                    className="header-center-content",
                    children=[
                        html.H1(
                            "Social Media Intelligence Hub",
                            style={
                                "margin": "0 0 6px 0",
                                "fontSize": "34px",
                                "fontWeight": "900",
                                "letterSpacing": "-0.8px",
                                "color": "#ffffff",
                                "textAlign": "center",
                                "textShadow": "0 1px 12px rgba(0, 0, 0, 0.25)"
                            }
                        ),
                        html.P(
                            "Executive performance analytics, audience behavior, and content optimization insights.",
                            style={
                                "margin": "0 0 8px 0",
                                "fontSize": "17px",
                                "fontWeight": "500",
                                "color": "#ffffff",
                                "textAlign": "center",
                                "textShadow": "0 1px 12px rgba(0, 0, 0, 0.25)"
                            }
                        ),
                        html.P(
                            "Group 3: Juliana Campopiano  |  Lia Li  |  Hua Qing Liu  |  Wenbin Chen",
                            style={
                                "margin": "0 0 16px 0",
                                "fontSize": "16px",
                                "fontWeight": "700",
                                "color": "#ffffff",
                                "textAlign": "center",
                                "textShadow": "0 1px 12px rgba(0, 0, 0, 0.25)"
                            }
                        ),
                        # Research Question card in frosted glass (Centered & Auto-expanding)
                        html.Div(
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "center",
                                "gap": "14px",
                                "backgroundColor": "rgba(255, 255, 255, 0.22)",
                                "backdropFilter": "blur(14px)",
                                "-webkitBackdropFilter": "blur(14px)",
                                "padding": "14px 28px",
                                "borderRadius": "16px",
                                "border": "1px solid rgba(255, 255, 255, 0.40)",
                                "boxShadow": "0 6px 24px rgba(0, 0, 0, 0.08)",
                                "width": "fit-content",
                                "maxWidth": "100%",
                                "margin": "0 auto"
                            },
                            children=[
                                html.Span(
                                    "Research Question:",
                                    style={
                                        "fontWeight": "800",
                                        "fontSize": "17px",
                                        "color": "#ffffff",
                                        "whiteSpace": "nowrap",
                                        "letterSpacing": "-0.2px",
                                        "textShadow": "0 1px 8px rgba(0, 0, 0, 0.20)"
                                    }
                                ),
                                html.Span(
                                    "How do social media platform, content type, and posting time influence post engagement rates across different social media platforms?",
                                    style={
                                        "fontSize": "17px",
                                        "fontWeight": "600",
                                        "color": "#ffffff",
                                        "lineHeight": "1.5",
                                        "textAlign": "center",
                                        "textShadow": "0 1px 10px rgba(0, 0, 0, 0.20)"
                                    }
                                )
                            ]
                        )
                    ]
                ),
                html.Div(
                    className="header-social-wrapper",
                    children=[
                        html.Div(
                            style={"display": "flex", "gap": "8px", "flexWrap": "wrap"},
                            children=[
                                html.Button("\U0001F44D Like", id="like-btn", n_clicks=0, style=social_btn_style(False), title="Like this dashboard"),
                                html.Button("\U0001F514 Subscribe", id="sub-btn", n_clicks=0, style=social_btn_style(False), title="Subscribe to this dashboard"),
                                html.Button("\U0001F4AC Comments", id="comment-toggle-btn", n_clicks=0, style=social_btn_style(False), title="Read and leave comments"),
                            ]
                        ),
                        html.Div(
                            id="comment-panel",
                            style=COMMENT_PANEL_STYLE,
                            children=[
                                html.Div(
                                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "10px"},
                                    children=[
                                        html.Span("Leave a comment", style={"fontWeight": "800", "fontSize": "16px", "color": "#8134AF"}),
                                        html.Button(
                                            "\u2715",
                                            id="close-comment-btn",
                                            n_clicks=0,
                                            title="Close",
                                            style={
                                                "background": "transparent",
                                                "border": "none",
                                                "color": "#6b6b80",
                                                "fontSize": "18px",
                                                "fontWeight": "700",
                                                "cursor": "pointer",
                                                "padding": "2px 8px",
                                                "borderRadius": "8px",
                                                "lineHeight": "1",
                                            }
                                        ),
                                    ]
                                ),
                                dcc.Input(id="comment-name", type="text", placeholder="Your name (optional)", maxLength=24, style=_COMMENT_INPUT_STYLE),
                                dcc.Textarea(id="comment-text", placeholder="What do you think of this dashboard?", maxLength=200, style={**_COMMENT_INPUT_STYLE, "height": "70px", "resize": "vertical"}),
                                html.Button(
                                    "Post comment",
                                    id="post-comment-btn",
                                    n_clicks=0,
                                    style={
                                        "background": "linear-gradient(135deg, #F58529 0%, #DD2A7B 50%, #8134AF 100%)",
                                        "color": "#ffffff",
                                        "border": "none",
                                        "borderRadius": "10px",
                                        "padding": "9px 18px",
                                        "fontSize": "14px",
                                        "fontWeight": "700",
                                        "cursor": "pointer",
                                    }
                                ),
                                html.Div(id="comments-list", style={"maxHeight": "230px", "overflowY": "auto", "marginTop": "12px"}),
                            ]
                        ),
                        dcc.Store(id="liked-store", storage_type="local"),
                        dcc.Store(id="heat-sel", data=None),
                        dcc.Store(id="bar-sel", data=None),
                        dcc.Store(id="subscribed-store", storage_type="local"),
                        dcc.Interval(id="social-interval", interval=5000, n_intervals=0),
                    ]
                ),
                html.Div(
                    className="header-action-btn-wrapper",
                    children=[
                        html.Button(
                            "Reset Filters",
                            id="reset-btn",
                            n_clicks=0,
                            style={
                                "backgroundColor": "rgba(255, 255, 255, 0.25)",
                                "color": "#ffffff",
                                "border": "1px solid rgba(255, 255, 255, 0.50)",
                                "borderRadius": "12px",
                                "padding": "10px 20px",
                                "fontSize": "15px",
                                "fontWeight": "700",
                                "cursor": "pointer",
                                "backdropFilter": "blur(14px)",
                                "-webkitBackdropFilter": "blur(14px)",
                                "boxShadow": "0 4px 16px rgba(0, 0, 0, 0.12)",
                                "textShadow": "0 1px 10px rgba(0, 0, 0, 0.25)",
                                "transition": "all 0.2s ease"
                            }
                        )
                    ]
                )
            ]
        ),

        # Main Content Wrapper
        html.Div(
            style={
                "maxWidth": "1400px",
                "margin": "0 auto",
                "position": "relative",
                "zIndex": 2,
                "padding": "16px 32px 48px 32px"
            },
            children=[
                # Interactive Control Panel Card
                html.Div(
                    style={**CARD_BOX, "marginBottom": "24px", "position": "relative", "zIndex": 50},
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "16px", "flexWrap": "wrap", "gap": "10px"},
                            children=[
                                html.Span("Interactive Filters & Metrics", style={"fontWeight": "800", "fontSize": "18px", "color": "#8134AF"}),
                                html.Div(
                                    style={"display": "flex", "alignItems": "center", "gap": "10px"},
                                    children=[
                                        html.Label("Analysis Metric:", style={"fontSize": "15px", "fontWeight": "700", "color": "#1f1f2e"}),
                                        dcc.RadioItems(
                                            id="metric-selector",
                                            options=[
                                                {"label": " Engagement Rate (%)", "value": "Engagement_Rate"},
                                                {"label": " Total Interactions", "value": "Interactions"},
                                                {"label": " Total Views", "value": "Views"},
                                            ],
                                            value="Engagement_Rate",
                                            inline=True,
                                            style={"fontSize": "15px", "color": "#1f1f2e"},
                                            inputStyle={"marginRight": "5px", "marginLeft": "12px", "cursor": "pointer"}
                                        )
                                    ]
                                )
                            ]
                        ),
                        # Filter Dropdowns Grid
                        html.Div(
                            style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(240px, 1fr))", "gap": "16px"},
                            children=[
                                html.Div([
                                    html.Label("Platform", style={"fontSize": "14px", "fontWeight": "700", "textTransform": "uppercase", "color": "#6b6b80", "marginBottom": "6px", "display": "block"}),
                                    dcc.Dropdown(
                                        id="platform-filter",
                                        options=[{"label": p, "value": p} for p in platforms],
                                        value=[],
                                        multi=True,
                                        placeholder="All platforms",
                                        style={"fontSize": "15px"}
                                    )
                                ]),
                                html.Div([
                                    html.Label("Category", style={"fontSize": "14px", "fontWeight": "700", "textTransform": "uppercase", "color": "#6b6b80", "marginBottom": "6px", "display": "block"}),
                                    dcc.Dropdown(
                                        id="category-filter",
                                        options=[{"label": c, "value": c} for c in categories],
                                        value=[],
                                        multi=True,
                                        placeholder="All categories",
                                        style={"fontSize": "15px"}
                                    )
                                ]),
                                html.Div([
                                    html.Label("Influencer Tier", style={"fontSize": "14px", "fontWeight": "700", "textTransform": "uppercase", "color": "#6b6b80", "marginBottom": "6px", "display": "block"}),
                                    dcc.Dropdown(
                                        id="tier-filter",
                                        options=[{"label": (t + " (" + TIER_RANGE_LABELS[t] + ")") if t in TIER_RANGE_LABELS else t, "value": t} for t in tiers],
                                        value=[],
                                        multi=True,
                                        placeholder="All influencer tiers",
                                        style={"fontSize": "15px"}
                                    )
                                ]),
                                html.Div([
                                    html.Label("Sentiment", style={"fontSize": "14px", "fontWeight": "700", "textTransform": "uppercase", "color": "#6b6b80", "marginBottom": "6px", "display": "block"}),
                                    dcc.Dropdown(
                                        id="sentiment-filter",
                                        options=[{"label": s, "value": s} for s in sentiments],
                                        value=[],
                                        multi=True,
                                        placeholder="All sentiments",
                                        style={"fontSize": "15px"}
                                    )
                                ]),
                            ]
                        )
                    ]
                ),

                # KPI Metric Cards Row
                html.Div(id="kpi-cards-row", style={"display": "flex", "gap": "18px", "flexWrap": "wrap", "marginBottom": "24px"}),

                # Navigation Tabs in Glass Container
                html.Div(
                    className="tabs-glass-wrapper",
                    children=[
                        dcc.Tabs(
                            id="tabs",
                            value="tab-overview",
                            style={"border": "none", "backgroundColor": "transparent"},
                            children=[
                                dcc.Tab(
                                    label="Platform & Posting Trends",
                                    value="tab-overview",
                                    className="custom-tab",
                                    selected_className="custom-tab--selected",
                                    children=[
                                        html.Div(
                                            style={"display": "grid", "gridTemplateColumns": "minmax(0, 1.25fr) minmax(0, 1fr)", "gap": "20px", "marginTop": "18px"},
                                            children=[
                                                # Top row, left: Social Media Performance by Platform (slightly wider than the heatmap)
                                                html.Div(style={**CARD_BOX, "gridColumn": "1", "gridRow": "1"}, children=[html.Div(style={"minHeight": "28px", "marginBottom": "4px"}), dcc.Graph(id="chart-platform", style={"height": "510px"}, config={"displayModeBar": False})]),
                                                # Top row, right: Best Posting Times Heatmap
                                                html.Div(style={**CARD_BOX, "gridColumn": "2", "gridRow": "1"}, children=[
                                                    html.Div(style={"display": "flex", "justifyContent": "flex-end", "minHeight": "28px", "marginBottom": "4px"}, children=[
                                                        html.Button("\u2715 Clear time filter", id="heat-clear-btn", n_clicks=0, style={"display": "none"}),
                                                        html.Div("\U0001F446 Click any cell to see specifics", id="heat-hint", style=HEAT_HINT_STYLE),
                                                    ]),
                                                    dcc.Graph(id="chart-heatmap", config={"displayModeBar": False}),
                                                ]),
                                                # Bottom: Engagement Rate over time, full width but a shorter strip
                                                html.Div(style={**CARD_BOX, "gridColumn": "1 / -1", "gridRow": "2"}, children=[dcc.Graph(id="chart-monthly-trend", style={"height": "320px"}, config={"displayModeBar": False})]),
                                            ]
                                        )
                                    ]
                                ),
                                dcc.Tab(
                                    label="Content Breakdown",
                                    value="tab-audience",
                                    className="custom-tab",
                                    selected_className="custom-tab--selected",
                                    children=[
                                        html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "gap": "12px", "marginBottom": "14px", "flexWrap": "wrap"}, children=[
                                            html.Div("\U0001F446 Click any bar in these charts to filter the others", id="bar-banner-text", style=BAR_BANNER_STYLE),
                                            html.Button("\u2715 Clear selection", id="bar-clear-btn", n_clicks=0, style={"display": "none"}),
                                        ]),
                                        html.Div(
                                            style={"display": "grid", "gridTemplateColumns": "minmax(0, 1.25fr) minmax(0, 1fr)", "gridTemplateRows": "auto auto", "gap": "20px", "marginTop": "18px"},
                                            children=[
                                                # Left: Performance by Content Type (large, spans both rows)
                                                html.Div(style={**CARD_BOX, "gridColumn": "1", "gridRow": "1"}, children=[dcc.Graph(id="chart-content-type", style={"height": "450px"}, config={"displayModeBar": False})]),
                                                # Right, top: Category Leaderboard
                                                html.Div(style={**CARD_BOX, "gridColumn": "2", "gridRow": "1"}, children=[dcc.Graph(id="chart-category", config={"displayModeBar": False})]),
                                                # Right, bottom: Influencer Tier Performance
                                                html.Div(style={**CARD_BOX, "gridColumn": "2", "gridRow": "2"}, children=[dcc.Graph(id="chart-tier", config={"displayModeBar": False})]),
                                                # Platform × Content Type
                                                html.Div(style={**CARD_BOX, "gridColumn": "1", "gridRow": "2"}, children=[dcc.Graph(id="chart-platform-content", style={"minHeight": "330px"}, config={"displayModeBar": False})]),
                                            ]
                                        )
                                    ]
                                ),
                                dcc.Tab(
                                    label="TikTok Video Spotlight",
                                    value="tab-video",
                                    className="custom-tab",
                                    selected_className="custom-tab--selected",
                                    children=[
                                        html.Div(
                                            style={**CARD_BOX, "marginTop": "18px"},
                                            children=[
                                                html.Div(
                                                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "20px", "flexWrap": "wrap", "gap": "14px"},
                                                    children=[
                                                        html.Div([
                                                            html.Div(
                                                                style={"display": "flex", "alignItems": "center", "marginBottom": "4px"},
                                                                children=[
                                                                    html.H3("TikTok Video Spotlight", style={"margin": 0, "fontSize": "20px", "color": "#8134AF", "fontWeight": "800"}),
                                                                    html.Span("Official TikTok Player", style={"fontSize": "13px", "fontWeight": "700", "backgroundColor": "rgba(221, 42, 123, 0.12)", "color": "#DD2A7B", "padding": "3px 8px", "borderRadius": "12px", "marginLeft": "10px"})
                                                                ]
                                                            ),
                                                            html.P("Watch featured TikTok videos and explore retention drivers and engagement metrics.", style={"margin": 0, "fontSize": "15px", "color": "#6b6b80"}),
                                                        ]),
                                                        html.Div(
                                                            style={"display": "flex", "alignItems": "center", "gap": "10px", "flexWrap": "wrap"},
                                                            children=[
                                                                dcc.Dropdown(
                                                                    id="video-selector",
                                                                    options=[
                                                                        {"label": v["title"], "value": k} for k, v in TIKTOK_VIDEOS_METADATA.items()
                                                                    ],
                                                                    value="6768504823336815877",
                                                                    clearable=False,
                                                                    style={"minWidth": "380px", "fontSize": "15px"}
                                                                ),
                                                            ]
                                                        )
                                                    ]
                                                ),
                                                # Video & Analytics 2-Column Grid
                                                html.Div(
                                                    style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(360px, 1fr))", "gap": "28px", "alignItems": "center"},
                                                    children=[
                                                        # Embedded Vertical TikTok Phone Player Frame
                                                        html.Div(
                                                            style={
                                                                "display": "flex",
                                                                "justifyContent": "center",
                                                                "alignItems": "center",
                                                                "backgroundColor": "#0f172a",
                                                                "borderRadius": "24px",
                                                                "padding": "20px 16px",
                                                                "boxShadow": "0 14px 40px rgba(221, 42, 123, 0.22)",
                                                                "border": "2px solid #DD2A7B"
                                                            },
                                                            children=[
                                                                html.Iframe(
                                                                    id="video-player-iframe",
                                                                    src="https://www.tiktok.com/player/v1/6768504823336815877?music_info=1&description=1",
                                                                    style={
                                                                        "width": "340px",
                                                                        "height": "600px",
                                                                        "borderRadius": "16px",
                                                                        "border": "none",
                                                                        "backgroundColor": "#000000"
                                                                    },
                                                                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share; fullscreen"
                                                                )
                                                            ]
                                                        ),
                                                        # Video Performance & Insights Sidebar
                                                        html.Div(
                                                            style={"display": "flex", "flexDirection": "column", "gap": "18px"},
                                                            children=[
                                                                # Metrics Row (Dynamically Updated via Callback)
                                                                html.Div(
                                                                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "14px"},
                                                                    children=[
                                                                        html.Div(
                                                                            style={"backgroundColor": "rgba(129, 52, 175, 0.14)", "padding": "16px", "borderRadius": "14px", "border": "1.5px solid rgba(129, 52, 175, 0.40)"},
                                                                            children=[
                                                                                html.Span("TOTAL TIKTOK VIEWS", style={"fontSize": "13px", "fontWeight": "700", "color": "#8134AF"}),
                                                                                html.Div("2.4B", id="tiktok-metric-views", style={"fontSize": "26px", "fontWeight": "800", "color": "#8134AF", "marginTop": "4px", "textShadow": "0 1px 2px rgba(129,52,175,0.15)"})
                                                                            ]
                                                                        ),
                                                                        html.Div(
                                                                            style={"backgroundColor": "rgba(129, 52, 175, 0.14)", "padding": "16px", "borderRadius": "14px", "border": "1.5px solid rgba(129, 52, 175, 0.40)"},
                                                                            children=[
                                                                                html.Span("ENGAGEMENT RATE", style={"fontSize": "13px", "fontWeight": "700", "color": "#DD2A7B"}),
                                                                                html.Div("1.3%", id="tiktok-metric-engagement", style={"fontSize": "26px", "fontWeight": "800", "color": "#DD2A7B", "marginTop": "4px"})
                                                                            ]
                                                                        ),
                                                                        html.Div(
                                                                            style={"backgroundColor": "rgba(129, 52, 175, 0.14)", "padding": "16px", "borderRadius": "14px", "border": "1.5px solid rgba(129, 52, 175, 0.40)"},
                                                                            children=[
                                                                                html.Span("TIKTOK SHARES", style={"fontSize": "13px", "fontWeight": "700", "color": "#F58529"}),
                                                                                html.Div("1.1M", id="tiktok-metric-shares", style={"fontSize": "26px", "fontWeight": "800", "color": "#F58529", "marginTop": "4px"})
                                                                            ]
                                                                        ),
                                                                        html.Div(
                                                                            style={"backgroundColor": "rgba(129, 52, 175, 0.14)", "padding": "16px", "borderRadius": "14px", "border": "1.5px solid rgba(129, 52, 175, 0.40)"},
                                                                            children=[
                                                                                html.Span("MEDIA FORMAT", style={"fontSize": "13px", "fontWeight": "700", "color": "#8134AF"}),
                                                                                html.Div("Magic Illusion", id="tiktok-metric-format", style={"fontSize": "17px", "fontWeight": "700", "color": "#8134AF", "marginTop": "6px"})
                                                                            ]
                                                                        ),
                                                                    ]
                                                                ),
                                                            ]
                                                        )
                                                    ]
                                                )
                                            ]
                                        )
                                    ]
                                ),
                                dcc.Tab(
                                    label="Top Posts Leaderboard",
                                    value="tab-posts",
                                    className="custom-tab",
                                    selected_className="custom-tab--selected",
                                    children=[
                                        html.Div(
                                            style={**CARD_BOX, "marginTop": "18px"},
                                            children=[
                                                html.Div(
                                                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "16px"},
                                                    children=[
                                                        html.H3("Top Performing Posts (Interactive Table)", style={"margin": 0, "fontSize": "19px", "color": "#8134AF", "fontWeight": "800"}),
                                                        html.Span("Sorted by Engagement Rate • Click any column header to re-sort", style={"fontSize": "15px", "color": "#6b6b80"})
                                                    ]
                                                ),
                                                dash_table.DataTable(
                                                    id="top-posts-table",
                                                    columns=[
                                                        {"name": "Post ID", "id": "Post_ID"},
                                                        {"name": "Platform", "id": "Platform"},
                                                        {"name": "Content Type", "id": "Content_Type"},
                                                        {"name": "Category", "id": "Category"},
                                                        {"name": "Tier", "id": "Influencer_Tier"},
                                                        {"name": "Engagement Rate (%)", "id": "Engagement_Rate", "type": "numeric"},
                                                        {"name": "Likes", "id": "Likes", "type": "numeric"},
                                                        {"name": "Comments", "id": "Comments", "type": "numeric"},
                                                        {"name": "Views", "id": "Views", "type": "numeric"},
                                                    ],
                                                    data=[],
                                                    sort_action="native",
                                                    page_action="none",
                                                    style_header={
                                                        "background": "linear-gradient(135deg, rgba(253, 242, 248, 0.95) 0%, rgba(255, 241, 242, 0.95) 100%)",
                                                        "fontWeight": "800",
                                                        "color": "#8134AF",
                                                        "border": "1px solid rgba(255, 255, 255, 0.6)",
                                                        "fontSize": "15px"
                                                    },
                                                    style_cell={
                                                        "padding": "12px 16px",
                                                        "fontSize": "15px",
                                                        "fontFamily": "inherit",
                                                        "textAlign": "left",
                                                        "color": "#1f1f2e",
                                                        "border": "1px solid rgba(255, 255, 255, 0.45)",
                                                        "backgroundColor": "rgba(255, 255, 255, 0.70)"
                                                    },
                                                    style_data_conditional=[
                                                        {"if": {"row_index": "odd"}, "backgroundColor": "rgba(255, 245, 248, 0.60)"},
                                                        {"if": {"column_id": "Engagement_Rate"}, "fontWeight": "bold", "color": "#DD2A7B"}
                                                    ]
                                                )
                                            ]
                                        )
                                    ]
                                ),
                            ]
                        )
                    ]
                )
            ]
        )
    ]
)

# -------------------------------------------------------------------
# 4. Helper Function: Number Formatter
# -------------------------------------------------------------------
def fmt(num):
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    if num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    if num >= 1_000:
        return f"{num / 1_000:.1f}K"
    return f"{num:,.0f}"

# -------------------------------------------------------------------
# 5. Reset Filters Callback
# -------------------------------------------------------------------
@app.callback(
    [
        Output("platform-filter", "value"),
        Output("category-filter", "value"),
        Output("tier-filter", "value"),
        Output("sentiment-filter", "value"),
        Output("metric-selector", "value")
    ],
    Input("reset-btn", "n_clicks"),
    prevent_initial_call=True
)
def reset_all_filters(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return [], [], [], [], "Engagement_Rate"

# -------------------------------------------------------------------
# 5a. Heatmap click -> cross-filter the Platform and Trend charts
# -------------------------------------------------------------------
HEAT_CLEAR_STYLE_SHOWN = {
    "backgroundColor": "rgba(129, 52, 175, 0.12)", "color": "#8134AF",
    "border": "1px solid rgba(129, 52, 175, 0.40)", "borderRadius": "999px",
    "padding": "4px 12px", "fontSize": "12px", "fontWeight": "700", "cursor": "pointer",
}


@app.callback(
    Output("heat-sel", "data"),
    [
        Input("chart-heatmap", "clickData"),
        Input("heat-clear-btn", "n_clicks"),
        Input("reset-btn", "n_clicks"),
    ],
    State("heat-sel", "data"),
    prevent_initial_call=True
)
def update_heat_selection(click, clear_clicks, reset_clicks, current):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""
    if trigger in ("heat-clear-btn", "reset-btn"):
        return None
    pts = (click or {}).get("points") or []
    if not pts:
        raise PreventUpdate
    x, y = pts[0].get("x"), pts[0].get("y")
    short = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hours = {f"{12 if h in (0, 12) else h % 12} {'AM' if h < 12 else 'PM'}": h for h in range(24)}
    if x not in short or y not in hours:
        raise PreventUpdate
    new = {"day": days_order[short.index(x)], "hour": hours[y], "x": x, "y": y}
    if current and current.get("day") == new["day"] and current.get("hour") == new["hour"]:
        return None
    return new


@app.callback(
    [Output("heat-clear-btn", "style"), Output("heat-hint", "style")],
    Input("heat-sel", "data"),
)
def show_heat_clear(sel):
    if sel:
        return HEAT_CLEAR_STYLE_SHOWN, {"display": "none"}
    return {"display": "none"}, HEAT_HINT_STYLE


# -------------------------------------------------------------------
# 5a-2. Bar clicks on tab 2 -> cross-filter Content Type / Category / Tier charts
# -------------------------------------------------------------------
@app.callback(
    Output("bar-sel", "data"),
    [
        Input("chart-content-type", "clickData"),
        Input("chart-category", "clickData"),
        Input("chart-tier", "clickData"),
        Input("bar-clear-btn", "n_clicks"),
        Input("reset-btn", "n_clicks"),
    ],
    State("bar-sel", "data"),
    prevent_initial_call=True
)
def update_bar_selection(content_click, category_click, tier_click, clear_clicks, reset_clicks, current):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""
    if trigger in ("bar-clear-btn", "reset-btn"):
        return None
    if trigger == "chart-content-type":
        key, click, axis = "content", content_click, "y"
    elif trigger == "chart-category":
        key, click, axis = "category", category_click, "x"
    elif trigger == "chart-tier":
        key, click, axis = "tier", tier_click, "x"
    else:
        raise PreventUpdate
    pts = (click or {}).get("points") or []
    val = pts[0].get(axis) if pts else None
    if not val:
        raise PreventUpdate
    cur = dict(current or {})
    cur[key] = None if cur.get(key) == val else val
    cur = {k: v for k, v in cur.items() if v}
    return cur or None


@app.callback(
    [Output("bar-banner-text", "children"), Output("bar-clear-btn", "style")],
    Input("bar-sel", "data"),
)
def show_bar_banner(sel):
    if not sel:
        return "\U0001F446 Click any bar in these charts to filter the others", {"display": "none"}
    labels = {"content": "Content Type", "category": "Category", "tier": "Tier"}
    parts = [labels[k] + ": " + str(sel[k]) for k in ("content", "category", "tier") if sel.get(k)]
    return "Filtering by  " + "  \u00b7  ".join(parts), HEAT_CLEAR_STYLE_SHOWN


# -------------------------------------------------------------------
# 5b. Video Player Source Callback
# -------------------------------------------------------------------
@app.callback(
    [
        Output("video-player-iframe", "src"),
        Output("tiktok-metric-views", "children"),
        Output("tiktok-metric-engagement", "children"),
        Output("tiktok-metric-shares", "children"),
        Output("tiktok-metric-format", "children"),
    ],
    Input("video-selector", "value"),
    prevent_initial_call=True
)
def update_video_src(preset_id):
    target_id = preset_id if preset_id in TIKTOK_VIDEOS_METADATA else "6768504823336815877"
    meta = TIKTOK_VIDEOS_METADATA[target_id]
    src = f"https://www.tiktok.com/player/v1/{target_id}?music_info=1&description=1"
    return (
        src,
        meta["views"],
        meta["engagement"],
        meta["shares"],
        meta["format"],
    )

# -------------------------------------------------------------------
# Social widget callbacks
# -------------------------------------------------------------------
@app.callback(
    Output("comment-panel", "style"),
    [
        Input("comment-toggle-btn", "n_clicks"),
        Input("close-comment-btn", "n_clicks"),
    ],
    State("comment-panel", "style"),
    prevent_initial_call=True
)
def toggle_comment_panel(toggle_clicks, close_clicks, style):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""
    style = dict(style or COMMENT_PANEL_STYLE)
    if trigger == "close-comment-btn":
        style["display"] = "none"
    else:
        style["display"] = "block" if style.get("display") == "none" else "none"
    return style


@app.callback(
    [
        Output("like-btn", "children"),
        Output("like-btn", "style"),
        Output("sub-btn", "children"),
        Output("sub-btn", "style"),
        Output("comment-toggle-btn", "children"),
        Output("comments-list", "children"),
        Output("liked-store", "data"),
        Output("subscribed-store", "data"),
        Output("comment-text", "value"),
    ],
    [
        Input("like-btn", "n_clicks"),
        Input("sub-btn", "n_clicks"),
        Input("post-comment-btn", "n_clicks"),
        Input("social-interval", "n_intervals"),
    ],
    [
        State("liked-store", "data"),
        State("subscribed-store", "data"),
        State("comment-name", "value"),
        State("comment-text", "value"),
    ]
)
def update_social(like_clicks, sub_clicks, post_clicks, n_intervals, liked, subscribed, name, text):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""
    liked, subscribed, clear_text, snap = apply_social_action(trigger, liked, subscribed, name, text)
    like_label = "\U0001F44D Liked" if liked else "\U0001F44D Like"
    sub_label = "\u2713 Subscribed" if subscribed else "\U0001F514 Subscribe"
    return (
        like_label + " \u00b7 " + compact_count(snap["likes"]),
        social_btn_style(liked),
        sub_label + " \u00b7 " + compact_count(snap["subscribers"]),
        social_btn_style(subscribed),
        "\U0001F4AC Comments \u00b7 " + compact_count(len(snap["comments"])),
        render_comments(snap["comments"]),
        liked,
        subscribed,
        "" if clear_text else dash.no_update,
    )


# -------------------------------------------------------------------
# 6. Main Dashboard Update Callback
# -------------------------------------------------------------------
@app.callback(
    [
        Output("kpi-cards-row", "children"),
        Output("chart-monthly-trend", "figure"),
        Output("chart-platform", "figure"),
        Output("chart-content-type", "figure"),
        Output("chart-heatmap", "figure"),
        Output("chart-tier", "figure"),
        Output("chart-category", "figure"),
        Output("chart-platform-content", "figure"),
        Output("top-posts-table", "data"),
    ],
    [
        Input("platform-filter", "value"),
        Input("category-filter", "value"),
        Input("tier-filter", "value"),
        Input("sentiment-filter", "value"),
        Input("metric-selector", "value"),
        Input("heat-sel", "data"),
        Input("bar-sel", "data"),
    ]
)
def update_dashboard(selected_platforms, selected_categories, selected_tiers, selected_sentiments, selected_metric, heat_sel=None, bar_sel=None):
    # Filter dataset
    filtered_df = df
    if selected_platforms:
        filtered_df = filtered_df[filtered_df["Platform"].isin(selected_platforms)]
    if selected_categories:
        filtered_df = filtered_df[filtered_df["Category"].isin(selected_categories)]
    if selected_tiers:
        filtered_df = filtered_df[filtered_df["Influencer_Tier"].isin(selected_tiers)]
    if selected_sentiments:
        filtered_df = filtered_df[filtered_df["Sentiment"].isin(selected_sentiments)]

    # A click on the Best Posting Times heatmap narrows only the Platform and Trend charts to that day + hour
    chart_df = filtered_df
    sel_suffix = ""
    if heat_sel and heat_sel.get("day") is not None and heat_sel.get("hour") is not None:
        chart_df = filtered_df[(filtered_df["Day_of_Week"] == heat_sel["day"]) & (filtered_df["Hour_of_Day"] == heat_sel["hour"])]
        sel_suffix = " <span style='font-size:13px;color:#6b6b80'>\u00b7 " + str(heat_sel.get("x", "")) + " " + str(heat_sel.get("y", "")) + "</span>"

    # Clicks on the Content Type / Category / Tier bars filter the OTHER two of those charts (a chart never filters itself)
    bar_sel = bar_sel or {}

    def narrow(base, skip):
        d = base
        if skip != "content" and bar_sel.get("content"):
            d = d[d["Content_Type"] == bar_sel["content"]]
        if skip != "category" and bar_sel.get("category"):
            d = d[d["Category"] == bar_sel["category"]]
        if skip != "tier" and bar_sel.get("tier"):
            d = d[d["Influencer_Tier"] == bar_sel["tier"]]
        return d

    content_src = narrow(filtered_df, "content")
    category_src = narrow(filtered_df, "category")
    tier_src = narrow(filtered_df, "tier")

    total_posts = len(filtered_df)
    total_dataset_size = len(df)
    pct_of_total = (total_posts / total_dataset_size * 100) if total_dataset_size > 0 else 0

    # Metrics computation
    avg_engagement = filtered_df["Engagement_Rate"].mean() if total_posts > 0 else 0
    total_views = filtered_df["Views"].sum() if total_posts > 0 else 0
    total_interactions = filtered_df["Interactions"].sum() if total_posts > 0 else 0

    # Metric label formatting
    metric_col = selected_metric
    is_rate = (selected_metric == "Engagement_Rate")
    metric_label = "Engagement Rate (%)" if is_rate else ("Total Interactions" if selected_metric == "Interactions" else "Total Views")
    agg_func = "mean" if is_rate else "sum"

    # KPI Cards Definition (Frosted Glass & Dark Secondary Text)
    kpi_cards = [
        html.Div(
            style={
                **CARD_BOX,
                "flex": "1",
                "minWidth": "220px",
                "borderLeft": "4px solid #DD2A7B"
            },
            children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                    children=[
                        html.Span("TOTAL POSTS", style={"fontSize": "13px", "fontWeight": "700", "color": "#DD2A7B", "letterSpacing": "0.5px"}),
                    ]
                ),
                html.Div(f"{total_posts:,}", style={"fontSize": "30px", "fontWeight": "800", "color": "#DD2A7B", "margin": "8px 0 4px 0"}),
            ]
        ),
        html.Div(
            style={
                **CARD_BOX,
                "flex": "1",
                "minWidth": "220px",
                "borderLeft": "4px solid #F58529"
            },
            children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                    children=[
                        html.Span("AVG ENGAGEMENT RATE", style={"fontSize": "13px", "fontWeight": "700", "color": "#F58529", "letterSpacing": "0.5px"}),
                    ]
                ),
                html.Div(f"{avg_engagement:.2f}%", style={"fontSize": "30px", "fontWeight": "800", "color": "#F58529", "margin": "8px 0 4px 0"}),
            ]
        ),
        html.Div(
            style={
                **CARD_BOX,
                "flex": "1",
                "minWidth": "220px",
                "borderLeft": "4px solid #8134AF"
            },
            children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                    children=[
                        html.Span("TOTAL VIEWS", style={"fontSize": "13px", "fontWeight": "700", "color": "#8134AF", "letterSpacing": "0.5px"}),
                    ]
                ),
                html.Div(fmt(total_views), style={"fontSize": "30px", "fontWeight": "800", "color": "#8134AF", "margin": "8px 0 4px 0"}),
            ]
        ),
        html.Div(
            style={
                **CARD_BOX,
                "flex": "1",
                "minWidth": "220px",
                "borderLeft": "4px solid #515BD4"
            },
            children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                    children=[
                        html.Span("TOTAL INTERACTIONS", style={"fontSize": "13px", "fontWeight": "700", "color": "#515BD4", "letterSpacing": "0.5px"}),
                    ]
                ),
                html.Div(fmt(total_interactions), style={"fontSize": "30px", "fontWeight": "800", "color": "#515BD4", "margin": "8px 0 4px 0"}),
            ]
        ),
    ]

    # Handle Empty State
    if total_posts == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            annotations=[dict(
                text="No posts match the current filter selection.<br>Try resetting or broadening your filters.",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="#8134AF")
            )]
        )
        return kpi_cards, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, []

    # ---------------------------------------------------------------
    # Chart 1: Monthly Trend (Smooth Area Line)
    # ---------------------------------------------------------------
    monthly = chart_df.groupby("Month", as_index=False)[metric_col].agg(agg_func)
    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Scatter(
        x=monthly["Month"], y=monthly[metric_col],
        mode="lines+markers",
        line=dict(color="#DD2A7B", width=3, shape="spline"),
        fill="tozeroy",
        fillcolor="rgba(221, 42, 123, 0.10)",
        marker=dict(size=8, color="#DD2A7B", line=dict(color="#ffffff", width=2)),
        hovertemplate=f"<b>Month</b>: %{{x}}<br><b>{metric_label}</b>: %{{y:,.2f}}<extra></extra>"
    ))
    fig_monthly.update_layout(
        title=dict(text=f"<b>Performance Over Time</b>{sel_suffix}", font=dict(color="#8134AF")),
        template="plotly_white",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=65, r=20, t=45, b=50),
        xaxis=dict(
            title=dict(text="<b>Month</b>", font=dict(color="#1f1f2e", size=13)),
            gridcolor="rgba(0, 0, 0, 0.06)",
            tickfont=dict(color="#1f1f2e")
        ),
        yaxis=dict(
            title=dict(text=f"<b>{metric_label}</b>", font=dict(color="#1f1f2e", size=13)),
            gridcolor="rgba(0, 0, 0, 0.06)",
            tickfont=dict(color="#1f1f2e")
        )
    )

    # ---------------------------------------------------------------
    # Chart 2: Platform Comparison with Official Brand Logos & Value Labels
    # ---------------------------------------------------------------
    plat = chart_df.groupby("Platform", as_index=False)[metric_col].agg(agg_func).sort_values(metric_col, ascending=False)
    plat_colors = [PLATFORM_COLORS.get(p, "#8134AF") for p in plat["Platform"]]
    max_val = plat[metric_col].max() if len(plat) > 0 and plat[metric_col].max() > 0 else 1

    # Numbers at the top of each bar (prominent, bold data labels)
    if is_rate:
        bar_numbers = [f"<b>{v:.2f}%</b>" for v in plat[metric_col]]
    else:
        bar_numbers = [f"<b>{fmt(v)}</b>" for v in plat[metric_col]]

    fig_plat = go.Figure(go.Bar(
        x=plat["Platform"],
        y=plat[metric_col],
        text=bar_numbers,
        textposition="outside",
        cliponaxis=False,
        textfont=dict(size=13, color="#0f172a", family="inherit"),
        marker=dict(color=plat_colors, line=dict(color="#ffffff", width=1.5)),
        hovertemplate=f"<b>%{{x}}</b><br>{metric_label}: %{{y:,.2f}}<extra></extra>"
    ))

    # Average threshold benchmark line across platforms (subtle background reference)
    if is_rate:
        # Same overall average as the Avg Engagement Rate card at the top (every post counts once)
        avg_val = chart_df[metric_col].mean() if len(chart_df) > 0 else 0
    else:
        avg_val = plat[metric_col].mean() if len(plat) > 0 else 0
    avg_str = f"{avg_val:.2f}%" if is_rate else fmt(avg_val)

    fig_plat.add_hline(
        y=avg_val,
        line_dash="dash",
        line_color="rgba(129, 52, 175, 0.38)",
        line_width=1.5,
        layer="below",
        annotation_text=f"Avg: {avg_str}",
        annotation_position="top right",
        annotation_font=dict(size=11, color="#6b6b80", family="inherit"),
        annotation_bgcolor="rgba(255, 255, 255, 0.75)",
        annotation_borderwidth=0
    )

    # Replace words on X-axis with official brand vector logos
    for _, row in plat.iterrows():
        p_name = row["Platform"]
        if p_name in PLATFORM_LOGOS_B64:
            fig_plat.add_layout_image(
                dict(
                    source=PLATFORM_LOGOS_B64[p_name],
                    xref="x",
                    yref="paper",
                    x=p_name,
                    y=-0.04,
                    sizex=0.52,
                    sizey=0.13,
                    xanchor="center",
                    yanchor="top"
                )
            )

    fig_plat.update_layout(
        title=dict(text=f"<b>Social Media Performance by Platform</b>{sel_suffix}", font=dict(color="#8134AF"), yref="container", y=0.9804, yanchor="middle"),
        template="plotly_white",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=65, r=20, t=90, b=110),
        xaxis=dict(
            title=dict(text="<b>Platform</b>", font=dict(color="#1f1f2e", size=13), standoff=70),
            tickmode="array",
            tickvals=plat["Platform"],
            ticktext=["" for _ in plat["Platform"]],
            ticks="",
            showgrid=False
        ),
        yaxis=dict(
            title=dict(text=f"<b>{metric_label}</b>", font=dict(color="#1f1f2e", size=13)),
            gridcolor="rgba(0, 0, 0, 0.06)",
            tickfont=dict(color="#1f1f2e"),
            range=[0, max_val * 1.18]
        )
    )

    # ---------------------------------------------------------------
    # Chart 3: Content Type Breakdown
    # ---------------------------------------------------------------
    content = content_src.groupby("Content_Type", as_index=False)[metric_col].agg(agg_func).sort_values(metric_col, ascending=True)
    bar_colors = ["#CFC8DC"] * len(content)
    if len(bar_colors) >= 3:
        bar_colors[-3] = "#DD2A7B"
        bar_colors[-2] = "#DD2A7B"
        bar_colors[-1] = "#DD2A7B"
    fig_content = go.Figure(go.Bar(
        y=content["Content_Type"], x=content[metric_col],
        orientation="h",
        marker=dict(color=bar_colors),
        selectedpoints=([list(content["Content_Type"]).index(bar_sel["content"])] if bar_sel.get("content") in list(content["Content_Type"]) else None),
        selected=dict(marker=dict(opacity=1)),
        unselected=dict(marker=dict(opacity=0.35)),
        hovertemplate=f"<b>%{{y}}</b><br>{metric_label}: %{{x:,.2f}}<extra></extra>"
    ))
    fig_content.update_layout(
        title=dict(text=f"<b>Performance by Content Type</b>", font=dict(color="#8134AF")),
        template="plotly_white",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=140, r=20, t=45, b=50),
        xaxis=dict(
            title=dict(text=f"<b>{metric_label}</b>", font=dict(color="#1f1f2e", size=13)),
            gridcolor="rgba(0, 0, 0, 0.06)",
            tickfont=dict(color="#1f1f2e")
        ),
        yaxis=dict(
            showticklabels=False
        ),
        annotations=[
            dict(xref="paper", yref="y", x=0, y=ct, xanchor="left", yanchor="middle", xshift=-112,
                 text=str(ct), showarrow=False, align="left", font=dict(color="#1f1f2e", size=12))
            for ct in content["Content_Type"]
        ] + [
            dict(xref="paper", yref="paper", x=0, y=0.5, xanchor="center", yanchor="middle", xshift=-130,
                 text="<b>Content Type</b>", textangle=-90, showarrow=False, font=dict(color="#1f1f2e", size=13))
        ]
    )

    # ---------------------------------------------------------------
    # Chart 4: Best Posting Times Heatmap (Day of Week vs Hour of Day)
    # ---------------------------------------------------------------
    heat_df = filtered_df.pivot_table(
        index="Hour_of_Day",
        columns="Day_of_Week",
        values=metric_col,
        aggfunc=agg_func
    ).reindex(index=range(24), columns=days_order)

    # Colour range follows the selected metric (Engagement Rate is a %, the others are counts in the thousands/millions).
    # Top of the scale is the 95th percentile so a few extreme cells don't wash out the rest.
    heat_finite = heat_df.values[np.isfinite(heat_df.values)]
    if heat_finite.size:
        heat_zmin = float(heat_finite.min())
        heat_zmax = float(np.percentile(heat_finite, 95))
        if heat_zmax <= heat_zmin:
            heat_zmax = heat_zmin + 1.0
    else:
        heat_zmin, heat_zmax = 0.0, 1.0
    heat_hover = "%{x} %{y} · %{z:.1f}%<extra></extra>" if is_rate else "%{x} %{y} · %{z:,.0f}<extra></extra>"

    hour_labels_12h = [f"{12 if h in (0, 12) else h % 12} {'AM' if h < 12 else 'PM'}" for h in range(24)]
    days_short = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    posting_heatmap_colorscale = [
        [0.0, "#FDE7CF"],
        [0.2, "#FBC98E"],
        [0.4, "#F58529"],
        [0.6, "#E8503F"],
        [0.8, "#DD2A7B"],
        [1.0, "#8134AF"]
    ]

    fig_heat = go.Figure(data=go.Heatmap(
        # plain nested lists (not a numpy array) so older Dash/plotly.js builds draw the cells instead of leaving them blank
        z=[[None if pd.isna(v) else float(v) for v in row] for row in heat_df.values],
        x=days_short,
        y=hour_labels_12h,
        colorscale=posting_heatmap_colorscale,
        zmin=heat_zmin,
        zmax=heat_zmax,
        xgap=2,
        ygap=2,
        hoverongaps=False,
        hovertemplate=heat_hover,
        colorbar=dict(
            thickness=14,
            len=1.0,
            y=0.5,
            yanchor="middle",
            tickformat=".0f" if is_rate else "~s",
            ticksuffix="%" if is_rate else "",
            tickfont=dict(color="#1f1f2e", size=10, family="inherit"),
            title=None
        )
    ))
    fig_heat.update_layout(
        title=dict(
            text=("<b>Best Posting Times Heatmap</b> <span style='font-size:12px;color:#6b6b80'>" + ("\u00b7 " + str(heat_sel.get("x", "")) + " " + str(heat_sel.get("y", "")) + " selected" if sel_suffix else "") + "</span>"),
            font=dict(color="#8134AF")
        ),
        meta=({"sel": [["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].index(heat_sel.get("x")), int(heat_sel.get("hour"))]} if sel_suffix and heat_sel.get("x") in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] else {}),
        template="plotly_white",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        height=510,
        margin=dict(l=48, r=15, t=32, b=28),
        xaxis=dict(
            title=None,
            tickmode="array",
            tickvals=days_short,
            ticktext=days_short,
            tickfont=dict(color="#1f1f2e", size=11, family="inherit"),
            tickangle=0,
            automargin=True
        ),
        yaxis=dict(
            title=None,
            autorange="reversed",
            tickmode="array",
            tickvals=[hour_labels_12h[h] for h in range(0, 24, 3)],
            ticktext=[hour_labels_12h[h] for h in range(0, 24, 3)],
            tickfont=dict(color="#1f1f2e", size=10, family="inherit"),
            automargin=True
        )
    )

    # ---------------------------------------------------------------
    # Chart 5: Influencer Tier Performance
    # ---------------------------------------------------------------
    tier_order = [t for t in tiers if t in tier_src["Influencer_Tier"].unique()]
    tier_df = tier_src.groupby("Influencer_Tier", as_index=False)[metric_col].agg(agg_func)
    tier_df["sort_order"] = tier_df["Influencer_Tier"].apply(lambda x: tier_order.index(x) if x in tier_order else 99)
    tier_df = tier_df.sort_values("sort_order")

    tier_palette = ["#FFDC80", "#FCAF45", "#F77737", "#E1306C", "#8134AF"]
    tier_bar_colors = [tier_palette[i % len(tier_palette)] for i in range(len(tier_df))]

    # Follower ranges per tier (taken from the dataset) so the tier names are self-explanatory
    tier_ranges = {"Nano": "under 10K", "Micro": "10K\u201350K", "Mid-tier": "50K\u2013100K", "Macro": "100K+"}
    tier_names = list(tier_df["Influencer_Tier"])
    fig_tier = go.Figure(go.Bar(
        x=tier_df["Influencer_Tier"], y=tier_df[metric_col],
        marker=dict(color=tier_bar_colors),
        customdata=[tier_ranges.get(n, "") for n in tier_names],
        selectedpoints=([tier_names.index(bar_sel["tier"])] if bar_sel.get("tier") in tier_names else None),
        selected=dict(marker=dict(opacity=1)),
        unselected=dict(marker=dict(opacity=0.35)),
        hovertemplate=f"<b>%{{x}} Tier</b> (%{{customdata}} followers)<br>{metric_label}: %{{y:,.2f}}<extra></extra>"
    ))
    fig_tier.update_layout(
        title=dict(text=f"<b>Audience Reach by Influencer Tier</b>", font=dict(color="#8134AF")),
        template="plotly_white",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=65, r=20, t=45, b=75),
        xaxis=dict(
            title=dict(text="<b>Influencer Tier</b> <span style='font-size:11px;color:#6b6b80'>(by follower count)</span>", font=dict(color="#1f1f2e", size=13)),
            tickfont=dict(color="#1f1f2e"),
            tickmode="array",
            tickvals=tier_names,
            ticktext=[n + "<br><span style='font-size:11px;color:#6b6b80'>" + tier_ranges[n] + "</span>" if n in tier_ranges else n for n in tier_names]
        ),
        yaxis=dict(
            title=dict(text=f"<b>{metric_label}</b>", font=dict(color="#1f1f2e", size=13)),
            gridcolor="rgba(0, 0, 0, 0.06)",
            tickfont=dict(color="#1f1f2e")
        )
    )

    # ---------------------------------------------------------------
    # Chart 7: Category Leaderboard (Top 8)
    # ---------------------------------------------------------------
    cat_df = category_src.groupby("Category", as_index=False)[metric_col].agg(agg_func).sort_values(metric_col, ascending=False).head(8)
    ig_cat_palette = ["#8134AF", "#C13584", "#DD2A7B", "#E1306C", "#FD1D1D", "#F77737", "#FCAF45", "#515BD4"]
    cat_bar_colors = [ig_cat_palette[i % len(ig_cat_palette)] for i in range(len(cat_df))]

    fig_cat = go.Figure(go.Bar(
        x=cat_df["Category"], y=cat_df[metric_col],
        marker=dict(color=cat_bar_colors),
        selectedpoints=([list(cat_df["Category"]).index(bar_sel["category"])] if bar_sel.get("category") in list(cat_df["Category"]) else None),
        selected=dict(marker=dict(opacity=1)),
        unselected=dict(marker=dict(opacity=0.35)),
        hovertemplate=f"<b>%{{x}}</b><br>{metric_label}: %{{y:,.2f}}<extra></extra>"
    ))
    fig_cat.update_layout(
        title=dict(text=f"<b>Top Categories Leaderboard</b>", font=dict(color="#8134AF")),
        template="plotly_white",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=65, r=20, t=45, b=50),
        xaxis=dict(
            title=dict(text="<b>Category</b>", font=dict(color="#1f1f2e", size=13)),
            tickfont=dict(color="#1f1f2e")
        ),
        yaxis=dict(
            title=dict(text=f"<b>{metric_label}</b>", font=dict(color="#1f1f2e", size=13)),
            gridcolor="rgba(0, 0, 0, 0.06)",
            tickfont=dict(color="#1f1f2e")
        )
    )

    # ---------------------------------------------------------------
    # Chart 8: Platform × Content Type Heatmap
    # ---------------------------------------------------------------
    if len(filtered_df) == 0:
        fig_plat_content = go.Figure()
        fig_plat_content.update_layout(
            title=dict(text=f"<b>Platform × Content Type</b>", font=dict(color="#8134AF")),
            template="plotly_white",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            height=330,
            margin=dict(l=85, r=20, t=40, b=85)
        )
    else:
        overall_content = (
            filtered_df.groupby("Content_Type")[metric_col]
            .agg(agg_func)
            .sort_values(ascending=False)
        )
        top_content_types = overall_content.head(8).index.tolist()

        overall_platform = (
            filtered_df.groupby("Platform")[metric_col]
            .agg(agg_func)
            .sort_values(ascending=False)
        )
        ordered_platforms = overall_platform.index.tolist()

        plat_content_pivot = filtered_df.pivot_table(
            index="Platform",
            columns="Content_Type",
            values=metric_col,
            aggfunc=agg_func
        ).reindex(index=ordered_platforms, columns=top_content_types)

        plat_vals = plat_content_pivot.values[np.isfinite(plat_content_pivot.values)]
        plat_zmax = 40 if is_rate else (float(np.percentile(plat_vals, 95)) if plat_vals.size else 1.0)
        if plat_zmax <= 0:
            plat_zmax = 1.0
        fig_plat_content = go.Figure(data=go.Heatmap(
            z=[[None if pd.isna(v) else float(v) for v in row] for row in plat_content_pivot.values],
            x=top_content_types,
            y=ordered_platforms,
            colorscale=posting_heatmap_colorscale,
            zmin=0,
            zmax=plat_zmax,
            xgap=3,
            ygap=3,
            hoverongaps=False,
            hovertemplate=("%{y} · %{x} · %{z:.1f}%<extra></extra>" if is_rate else "%{y} · %{x} · %{z:,.0f}<extra></extra>"),
            colorbar=dict(
                orientation="h",
                thickness=8,
                len=0.45,
                x=0.5,
                xanchor="center",
                y=-0.36,
                tickmode="array",
                tickvals=[0, 40],
                ticktext=["low", "40%+"],
                ticklen=0,
                tickfont=dict(color="#6b6b80", size=10, family="inherit"),
                outlinewidth=0,
                title=None
            )
        ))
        fig_plat_content.update_layout(
            title=dict(
                text=f"<b>Platform × Content Type</b>",
                font=dict(color="#8134AF")
            ),
            template="plotly_white",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            height=330,
            margin=dict(l=85, r=20, t=40, b=85),
            xaxis=dict(
                title=dict(text="<b>Content Type</b>", font=dict(color="#1f1f2e", size=13), standoff=8),
                tickmode="array",
                tickvals=top_content_types,
                ticktext=top_content_types,
                tickfont=dict(color="#1f1f2e", size=10, family="inherit"),
                tickangle=0,
                side="bottom",
                automargin=True
            ),
            yaxis=dict(
                title=dict(text="<b>Platform</b>", font=dict(color="#1f1f2e", size=13)),
                autorange="reversed",
                tickmode="array",
                tickvals=ordered_platforms,
                ticktext=ordered_platforms,
                tickfont=dict(color="#1f1f2e", size=11, family="inherit"),
                automargin=True
            )
        )

    # ---------------------------------------------------------------
    # Table: Top 15 Posts
    # ---------------------------------------------------------------
    top_posts = filtered_df.sort_values(by="Engagement_Rate", ascending=False).head(15)
    table_records = []
    for _, row in top_posts.iterrows():
        table_records.append({
            "Post_ID": row["Post_ID"],
            "Platform": row["Platform"],
            "Content_Type": row["Content_Type"],
            "Category": row["Category"],
            "Influencer_Tier": row["Influencer_Tier"],
            "Engagement_Rate": round(float(row["Engagement_Rate"]), 2),
            "Likes": f"{int(row['Likes']):,}",
            "Comments": f"{int(row['Comments']):,}",
            "Views": f"{int(row['Views']):,}",
        })

    return (
        kpi_cards,
        fig_monthly,
        fig_plat,
        fig_content,
        fig_heat,
        fig_tier,
        fig_cat,
        fig_plat_content,
        table_records
    )

# -------------------------------------------------------------------
# 7. Server Execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    print(f"Dashboard running at http://127.0.0.1:{port}/")
    if hasattr(app, "run"):
        app.run(debug=True, port=port)
    else:
        app.run_server(debug=True, port=port)
