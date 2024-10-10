import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tabulate import tabulate

from SpotifyAnalysis import SpotifyAnalysis

sns.set(style="whitegrid")
pd.set_option('display.max_columns', None)
pd.set_option("mode.chained_assignment", None)


# STREAM ANALYSIS
def get_streams(profile=None, entries=None, entry_type="artist", start_time=None,
                end_time=None, min_listen_time=None, first_listen=None, first_listen_type="artist",
                first_listen_tolerance=0):
    profile, start_time, end_time, min_listen_time = set_params(profile, start_time, end_time, min_listen_time)
    df = spotify.get_profile(profile)
    df = df.loc[:end_time, :]
    if first_listen:
        first_listen = pd.to_datetime(first_listen).tz_localize('UTC')
        df["group"] = df[first_listen_type] + df["artist"]
        entry_first_listen = df.groupby("group").apply(lambda x: x.index[min(first_listen_tolerance, len(x.index) - 1)])
        df = df[df.apply(lambda row: entry_first_listen.loc[row["group"]] >= first_listen, axis=1)]
        df = df.drop("group", axis=1)

    if entries:
        df = df[df[entry_type].str.lower().isin([x.lower() for x in entries])]
    df = df[df["ms"] >= min_listen_time]
    return df.loc[start_time:end_time]


def top_n(profile=None, df=None, entry_type="artist", master_entries=None, master_entry_type="artist", n=None,
          sort="freq", start_time=None, end_time=None, min_listen_time=None,
          descending=True, first_listen=None, first_listen_type="artist", first_listen_tolerance=0):
    profile, start_time, end_time, min_listen_time = set_params(profile, start_time, end_time, min_listen_time)
    if df is None:
        df = get_streams(profile, entries=master_entries, entry_type=master_entry_type, start_time=start_time,
                         end_time=end_time, min_listen_time=min_listen_time, first_listen=first_listen,
                         first_listen_type=first_listen_type, first_listen_tolerance=first_listen_tolerance)
    df["group"] = df[entry_type] + df["artist"]
    df = df.groupby("group")
    if entry_type == "track":
        df = df.agg(
            {"track": "first", "album": "first", "artist": "first", "freq": "sum", "ms": "sum", "skipped": "sum"})
    elif entry_type == "album":
        df = df.agg({"album": "first", "artist": "first", "freq": "sum", "ms": "sum", "skipped": "sum"})
    else:
        df = df.agg({"artist": "first", "freq": "sum", "ms": "sum", "skipped": "sum"})
    df["skip_ratio"] = df["skipped"] / df["freq"]
    if not n:
        n = len(df)
    if not descending:
        return df.nsmallest(n, sort).reset_index().drop("group", axis=1)
    return df.nlargest(n, sort).reset_index().drop("group", axis=1)


def get_totals(profile=None, df=None, entries=None, entry_type="artist", start_time=None, end_time=None,
               min_listen_time=None):
    profile, start_time, end_time, min_listen_time = set_params(profile, start_time, end_time, min_listen_time)
    if df is None:
        df = get_streams(profile, entries=entries, entry_type=entry_type, start_time=start_time, end_time=end_time,
                         min_listen_time=min_listen_time)
    freq = df["freq"].sum()
    ms = df["ms"].sum()
    minutes = int(ms / 60000)
    skipped = df["skipped"].sum()
    skip_ratio = skipped / freq
    return {"freq": freq, "ms": ms, "minutes": minutes, "skip_ratio": skip_ratio, "skipped": skipped}


def nth_occurrence(profile=None, df=None, entry=None, entry_type="artist", n=0, min_listen_time=None):
    """Assumes sorted timeseries df"""
    profile, _, _, min_listen_time = set_params(profile, None, None, min_listen_time)
    if df is None:
        df = get_streams(profile, min_listen_time=min_listen_time)
    if entry:
        df = df[df[entry_type].str.lower() == entry.lower()]
    return df.index[n], df.loc[df.index[n]]


# PLOTTING
def ts_plot(data, sort, sample_width, hue=None):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="ts", y=sort, hue=hue)
    if sort == "freq":
        label = 'Number of streams per '
    elif sort == "ms":
        plt.yticks(ticks=plt.yticks()[0], labels=[f'{int(y / 60000)}' for y in plt.yticks()[0]])
        label = 'Minutes listened per '
    else:
        label = "Percent of songs skipped per "
    plt.ylabel(label + date_width_str(sample_width))

    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title=(type+"s").capitalize(), title_fontsize='13')
    if hue is not None:
        plt.legend(title=(hue + "s").capitalize())
    plt.xlabel('Date')
    plt.xticks(rotation=-45)
    plt.grid(which='major', axis='x', color='gray', linestyle='-', linewidth=0.5)
    plt.show()


def plot_over_time(profile=None, df=None, entries=None, entry_type="artist", start_time=None, end_time=None,
                   min_listen_time=None, sort="freq", sample_width=None,
                   window_size=0):
    profile, start_time, end_time, min_listen_time = set_params(profile, start_time, end_time, min_listen_time)
    if df is None:
        df = get_streams(profile, start_time=start_time, end_time=end_time, min_listen_time=min_listen_time)
    sample_width = "D" if not sample_width and window_size else "2W" if not sample_width else sample_width
    resample = pd.DataFrame()
    if entries:
        for entry in entries:
            artist_data = df[df[entry_type].str.lower() == entry.lower()].sort_index()
            entry = artist_data[entry_type].iloc[0]
            artist_data = artist_data.resample(sample_width).sum()
            artist_data["skip_ratio"] = artist_data["skipped"] / artist_data["freq"]
            if window_size:
                artist_data = artist_data[sort].rolling(window=window_size).mean()
            if sort == "skip_ratio":
                artist_data = artist_data[artist_data["freq"] > 0]
            artist_data = artist_data.reset_index()
            artist_data[entry_type] = entry
            resample = pd.concat([resample, artist_data], ignore_index=True)
        ts_plot(resample, sort, sample_width, entry_type)
    else:
        resample = df.resample(sample_width).sum()
        resample["skip_ratio"] = resample["skipped"] / resample["freq"]
        if sort == "skip_ratio":
            resample = resample[resample["freq"] > 0]
        if window_size:
            resample = resample[sort].rolling(window=window_size).mean()
        ts_plot(resample.reset_index(), sort, sample_width)


# HELPERS
def set_params(profile=None, start_time=None, end_time=None, min_listen_time=None):
    if profile is None:
        profile = params["current_user"]
    if start_time is None:
        start_time = params["start"]
    if end_time is None:
        end_time = params["end"]
    if min_listen_time is None:
        min_listen_time = params["min_listen_time"]
    return profile, start_time, end_time, min_listen_time


# CLIENT METHODS
def streaming_overview(profile=None, sort="freq", n=5, detailed=False, discovery=False, entry_types=None,
                       start_time=None, end_time=None, min_listen_time=None, first_listen_tolerance=0):
    profile, start_time, end_time, min_listen_time = set_params(profile, start_time, end_time, min_listen_time)
    sub_totals = get_totals(profile, start_time=start_time, end_time=end_time, min_listen_time=min_listen_time)
    if detailed:
        end = " (Unfiltered Total: " + str(spotify.get_size(profile)) + ")\n"
    else:
        end = "\n"
    print("Total Streams: " + str(sub_totals["freq"]), end=end)
    print("Minutes Listened: " + str(sub_totals["minutes"]))
    print("Percent Skipped: " + str(int(round(sub_totals["skip_ratio"], 2) * 100)) + "%")

    if detailed:
        first = nth_occurrence(profile=profile, n=1, min_listen_time=min_listen_time)[0]
        last = nth_occurrence(profile=profile, n=-1, min_listen_time=min_listen_time)[0]
        print("First Stream on", str(first.date()), "at", f"{first.hour}:{first.minute}")
        print("Last Stream on", str(last.date()), "at", f"{last.hour}:{last.minute}")

    sort2 = "minutes" if sort == "ms" else sort
    first_listen = start_time if discovery else None
    if entry_types is not None:
        for i, entry_type in enumerate(entry_types):
            entry_types[i] = tabulate_top_n(
                top_n(profile, n=n, entry_type=entry_type, start_time=start_time, end_time=end_time,
                      first_listen=first_listen, first_listen_type=entry_type,
                      first_listen_tolerance=first_listen_tolerance, sort=sort), [entry_type, sort2])
        print(horizontal_concatenation(entry_types, gap="\t"))


def wrapped_redux(profile=None, sort="freq", n=5, discovery=False, first_listen_tolerance=0):
    profile, _, _, _ = set_params(profile, None, None, None)
    for year in spotify.get_years(profile):
        print("Year: " + str(year))
        streaming_overview(profile, sort, n, start_time=str(year), end_time=str(year), discovery=discovery,
                           entry_types=["track", "album", "artist"], first_listen_tolerance=first_listen_tolerance)


def tabulate_top_n_discovery(profile=None, entry_type="artist", sort="freq", n=10, start_time=None, end_time=None,
                             comparison_mode=True):
    if profile is None:
        profile = params["current_user"]
    top_new = tabulate_top_n(top_n(profile, n=n, entry_type=entry_type, first_listen=start_time, end_time=end_time,
                                   first_listen_type=entry_type), [entry_type, sort])
    if not comparison_mode:
        return top_new
    top = tabulate_top_n(top_n(n=n, entry_type=entry_type, start_time=start_time, end_time=end_time),
                         [entry_type, sort])
    return horizontal_concatenation([top_new, top])


# FORMATTING
def date_width_str(date_length):
    if date_length == "D":
        return "day"
    elif "D" in date_length:
        days = int(date_length.replace("D", ""))
        if days == 1:
            return "day"
        else:
            return f"{days} days"
    elif date_length == "W":
        return "week"
    elif "W" in date_length:
        weeks = int(date_length.replace("W", ""))
        if weeks == 1:
            return "week"
        else:
            return f"{weeks} weeks"
    elif date_length == "M":
        return "month"
    elif "M" in date_length:
        months = int(date_length.replace("M", ""))
        if months == 1:
            return "month"
        else:
            return f"{months} months"
    elif date_length == "Y":
        return "year"
    elif "Y" in date_length:
        years = int(date_length.replace("Y", ""))
        if years == 1:
            return "year"
        else:
            return f"{years} years"
    else:
        return "Invalid date length format"


def clean_headers(headers):
    for i in range(len(headers)):
        if headers[i] == "freq":
            headers[i] = "frequency"
        if headers[i] == "ms":
            headers[i] = "milliseconds"
        if headers[i] == "skip_ratio":
            headers[i] = "skipped ratio"
        headers[i] = headers[i].capitalize()
    return headers


def tabulate_top_n(df, cols=None, tablefmt="psql"):
    df["minutes"] = (df["ms"] / 60000).astype(int)
    if cols:
        cols = [x for x in cols if x in df.columns]
        df = df.loc[:, cols]
    df.index = df.index + 1
    return tabulate(df, headers=clean_headers(df.columns.tolist()), tablefmt=tablefmt)


def horizontal_concatenation(strings, gap=""):
    lines = [s.splitlines() for s in strings]
    max_lines = max(len(line) for line in lines)
    lines = [line + [''] * (max_lines - len(line)) for line in lines]
    horizontal_lines = [gap.join(line) for line in zip(*lines)]
    horizontal_result = '\n'.join(horizontal_lines)
    return horizontal_result


def format_integer_to_ordinal(number):
    suffix = "th" if abs(number) % 100 in [11, 12, 13] else {1: "st", 2: "nd", 3: "rd"}.get(abs(number) % 10, "th")
    return f"{number}{suffix}"


# UI HELPERS
def print_title(s, title_length):
    print(s, end='')
    for i in range(title_length - len(s)):
        print("-", end='')
    print()


def force_int(prompt="", allow_none=False, default=None, min_int=float("-inf"), max_int=float("inf"), display=False):
    if display and (default is not None or min_int != float("-inf") or max_int != float("inf")):
        prompt += " ("
        if default is not None:
            prompt += f"Default: {default}"
        if default is not None and (min_int != float("-inf") or max_int != float("inf")):
            prompt += " | "
        if min_int != float("-inf"):
            prompt += f"Min: {min_int}"
        if min_int != float("-inf") and max_int != float("inf"):
            prompt += " | "
        if max_int != float("inf"):
            prompt += f"Max: {max_int}"
        prompt += ")"
    while True:
        try:
            line = input(prompt + " ")
            if line == "" and (allow_none or default is not None):
                return None if allow_none else default
            line = int(line)
            if min_int <= line <= max_int:
                return line
            print("Please enter a number in the given range.")
        except ValueError:
            print("Please enter a valid integer.")


def force_date(prompt="", min_date=None, max_date=None, display=False):
    if display and (min_date is not None or max_date is not None):
        prompt += " ("
        if min_date is not None:
            prompt += f"Min: {min_date}"
        if min_date is not None and max_date is not None:
            prompt += " | "
        if max_date is not None:
            prompt += f"Max: {max_date}"
        prompt += ")"
    while True:
        try:
            line = input(prompt + " (M/D/Y)")
            if line.lower() == "none" or line == "":
                return None
            date = pd.to_datetime(line).date()
            min_date = pd.to_datetime(min_date).date() if min_date else min_date
            max_date = pd.to_datetime(max_date).date() if max_date else max_date
            if (not min_date or date >= min_date) and (not max_date or date <= max_date):
                return date.strftime("%m/%d/%Y")
            print("Please enter a date in the given range (M/D/Y) ")
        except ValueError:
            print("Please enter a valid date (M/D/Y) ")


def list_options(options, allow_none=False):
    for i, option in enumerate(options):
        print(f"{i + 1}. {option}")
    result = force_int("Enter your choice:", allow_none, min_int=1, max_int=len(options))
    return None if result is None else result - 1


def select_sort():
    print("Would you like results to be sorted based on stream frequency, total playtime, or skip ratio? (f/p/s)")
    line = input()
    while line.lower() not in ['f', 'p', 's']:
        print("Please provide a valid response")
        line = input()
    return {"f": "freq", "p": "ms", "s": "skip_ratio"}[line.lower()]


def select_entry_type(allow_none=False):
    entry_types = ["Track", "Album", "Artist"]
    if allow_none:
        entry_types.append("None")
    entry_type = entry_types[list_options(entry_types)].lower()
    return None if entry_type == "none" else entry_type


def yes_or_no(prompt=""):
    print(prompt + " (y/n)")
    line = input().lower()
    while line not in ['y', 'n']:
        print("Please provide a valid response")
        line = input()
    return line == 'y'


def select_user():
    users = spotify.get_users()
    print("Please Select a Profile")
    params["current_user"] = users[list_options(users)]


def select_entry(entry_type, prompt=False, allow_none=False):
    df = get_streams()
    while True:
        entry = input(("Please enter a streamed " + entry_type + ": ") if prompt else "").lower()
        if entry == "" and allow_none:
            return None
        artist_data = df[df[entry_type].str.lower() == entry.lower()]
        entries = artist_data[entry_type].unique()
        if len(entries) == 1:
            return entries[0]
        elif len(entries) > 1:
            print("MULTIPLE ENTRIES FUCKK:")
            print(entries)
        else:
            print(entry_type + " not found")


def select_entries(entry_type, ask_for_top=True):
    if ask_for_top:
        if yes_or_no("Would you like to get entries from a custom top n chart?"):
            [master_entry_type, master_entries, n, sort, start_time, end_time, first_listen, first_listen_type,
             tolerance] = \
                list_function_params("Top " + entry_type + "s Chart Creator",
                                     ["master_entry_type", "master_entries", "n", "sort", "start_time",
                                      "end_time", "first_listen", "first_listen_type", "tolerance"],
                                     [None, None, 10, "freq", None, None, None, "track", 0])
            print()
            return top_n(entry_type=entry_type, master_entry_type=master_entry_type,
                         master_entries=master_entries, n=n, sort=sort, start_time=start_time,
                         end_time=end_time, first_listen=first_listen, first_listen_type=first_listen_type,
                         first_listen_tolerance=tolerance)[entry_type].tolist()
    print("Enter", entry_type + "s, pressing enter between each entry. Press enter when finished")
    entries = set()
    while True:
        print(len(entries), "selected.", end=" ")
        entry = select_entry(entry_type, prompt=True, allow_none=True)
        if entry is None:
            break
        entries.add(entry)
    return list(entries)


def list_function_params(prompt, func_params, results):
    while True:
        options = []
        index = []
        for i, param in enumerate(func_params):
            index.append(i)
            if param == "entry_type":
                options.append("Entry Type: " + str(results[i]))
            elif param == "entry" and results[func_params.index("entry_type")]:
                options.append(results[func_params.index("entry_type")].capitalize() + ": " + str(results[i]))
            elif param == "entries" and results[func_params.index("entry_type")]:
                options.append(results[func_params.index("entry_type")].capitalize() + "s: " + str(results[i]))
            elif param == "master_entry_type":
                options.append("Master Entry Type: " + str(results[i]))
            elif param == "master_entries" and results[func_params.index("master_entry_type")] is not None:
                options.append("Master Entries: " + str(results[i]))
            elif param == "sort":
                options.append("Sort: " + results[i])
            elif param == "n":
                options.append("Entry Count: " + str(results[i]))
            elif param == "start_time":
                options.append("Start Date: " + str(results[i]))
            elif param == "end_time":
                options.append("End Date: " + str(results[i]))
            elif param == "index":
                options.append("Index: " + str(results[i]))
            elif param == "first_listen_type" and results[func_params.index("first_listen")] is not None:
                options.append("First listen filter type: " + str(results[i]))
            elif param == "first_listen":
                options.append("First listen filter date: " + str(results[i]))
            elif param == "discovery":
                options.append("Discovery mode: " + str(results[i]))
            elif param == "tolerance" and \
                    (("discovery" in func_params and results[func_params.index("discovery")]) or
                     ("first_listen" in func_params and results[func_params.index("first_listen")])):
                options.append("First Listen Tolerance: " + str(results[i]))
            elif param == "sample_width":
                options.append("Sample Width: " + ("Default" if results[i] is None else date_width_str(results[i])))
            elif param == "window_size":
                options.append("Rolling Window Size: " + str(results[i]))
            else:
                index.pop()
        print(prompt + ": Please choose parameters then press enter")

        choice = list_options(options, allow_none=True)
        if choice is None:
            return results
        i = index[choice]
        choice = func_params[i]
        if choice == "entry_type":
            results[i] = select_entry_type(allow_none=True)
            if "entry" in func_params:
                results[func_params.index("entry")] = None
        elif choice == "entry":
            results[i] = select_entry(results[func_params.index("entry_type")])
        elif choice == "entries":
            results[i] = select_entries(results[func_params.index("entry_type")])
        elif choice == "master_entry_type":
            results[i] = select_entry_type(allow_none=True)
            results[func_params.index("master_entries")] = None
        elif choice == "master_entries":
            results[i] = select_entries(results[func_params.index("master_entry_type")])
        elif choice == "sort":
            results[i] = select_sort()
        elif choice == "n":
            results[i] = force_int("How many entries per chart?", default=5, min_int=0, display=True)
        elif choice == "start_time":
            results[i] = \
                force_date("Enter a starting date:", display=True,
                           min_date=str(nth_occurrence()[0].date()) if params["start"] is None else params["start"],
                           max_date=str(nth_occurrence(n=-1)[0].date()) if params["end"] is None else params["end"])
        elif choice == "end_time":
            results[i] = \
                force_date("Enter a ending date:", display=True,
                           min_date=str(nth_occurrence()[0].date()) if params["start"] is None else params["start"],
                           max_date=str(nth_occurrence(n=-1)[0].date()) if params["end"] is None else params["end"])
        elif choice == "index":
            size = len(get_streams(entry_type=results[func_params.index("entry_type")],
                                   entries=[results[func_params.index("entry")]])) \
                if results[func_params.index("entry")] is not None else len(get_streams())
            results[i] = force_int("What stream index do you want?", default=0, min_int=-size + 1, max_int=size - 1,
                                   display=True)
        elif choice == "first_listen_type":
            results[i] = select_entry_type(allow_none=False)
        elif choice == "first_listen":
            results[i] = force_date("Select a first listen date:")
        elif choice == "discovery":
            results[i] = yes_or_no("Would you like to use discovery mode?")
        elif choice == "tolerance":
            results[i] = force_int("Enter a listening tolerance for discover mode:", default=0, min_int=0, display=True)
        elif choice == "sample_width":
            results[i] = input("Enter a sample width: (e.g. '2D', 'W, '1M')")
        elif choice == "window_size":
            results[i] = force_int("Enter a rolling window size:", allow_none=False, default=0, min_int=0)


# UI BRANCHES
def statistics_and_summaries():
    selection = 0
    while selection >= 0:
        print_title("Statistics and Summaries - Please select an option:", 70)
        selection = list_options(["Streaming History Overview", "Wrapped Redux", "First Occurrences", "Exit"])
        if selection == 0:
            print()
            streaming_overview(entry_types=["track", "album", "artist"], n=10, detailed=True)
        elif selection == 1:
            sort, n, d, t = list_function_params("Wrapped Redux",
                                                 ["sort", "n", "discovery", "tolerance"],
                                                 ["freq", 5, False, 0])
            print()
            wrapped_redux(sort=sort, n=n, discovery=d, first_listen_tolerance=t)
        elif selection == 2:
            entry_type, entry, i = list_function_params("First Occurrence", ["entry_type", "entry", "index"],
                                                        ["artist", None, 0])
            print()
            datetime, stream = nth_occurrence(entry=entry, entry_type=entry_type, n=i)
            date = str(datetime.date())
            time = f"{datetime.hour}:{datetime.minute}"
            i = format_integer_to_ordinal(i + (1 if i >= 0 else 0))
            if entry_type != "artist":
                print(stream["track"], "by", stream["artist"], "was streamed for the", i, "time on:", date, "at", time)
            else:
                print(stream["artist"], "was streamed:", date, "at", time)
            if entry_type != "track":
                print(i, "track streamed was:", stream["track"])
        else:
            selection = -1
        print()


def create_charts():
    [entry_type, master_entry_type, master_entries, n, sort, start_time, end_time, first_listen, first_listen_type,
     tolerance] = list_function_params("Chart Creator",
                                       ["entry_type", "master_entry_type", "master_entries", "n", "sort",
                                        "start_time", "end_time", "first_listen", "first_listen_type", "tolerance"],
                                       ["track", None, None, 10, "freq", None, None, None, "track", 0])
    print()
    top = top_n(entry_type=entry_type, master_entry_type=master_entry_type, master_entries=master_entries, n=n,
                sort=sort, start_time=start_time, end_time=end_time, first_listen=first_listen,
                first_listen_type=first_listen_type, first_listen_tolerance=tolerance)
    print(tabulate_top_n(top, cols=["track", "album", "artist", "freq", "minutes", "skip_ratio"]))


def create_plots():
    [entry_type, entries, sort, sample_width, window_size, start_time,
     end_time] = list_function_params("Plot Creator",
                                      ["entry_type", "entries", "sort", "sample_width",
                                       "window_size", "start_time", "end_time"],
                                      [None, None, "freq", None, 0, None, None])
    plot_over_time(entry_type=entry_type, entries=entries, sort=sort, sample_width=sample_width,
                   window_size=window_size, start_time=start_time, end_time=end_time)


def settings():
    selection = 0
    while selection >= 0:
        print_title("Settings - Please select an option:", 70)
        selection = list_options([f"Change User (Current: {params['current_user']})",
                                  f"Set Minimum Date (Current: {params['start']})",
                                  f"Set Maximum Date (Current: {params['end']})",
                                  f"Set Minimum Listen Time (Current: {params['min_listen_time']})", "Exit"])
        if selection == 0:
            select_user()
        elif selection == 1:
            params["start"] = force_date("Please enter a minimum date (Press Enter to remove filter)",
                                         min_date=str(nth_occurrence()[0].date()),
                                         max_date=str(nth_occurrence(n=-1)[0].date()), display=True)
        elif selection == 2:
            params["end"] = force_date("Please enter a maximum date (Press Enter to remove filter)",
                                       min_date=str(nth_occurrence()[0].date()),
                                       max_date=str(nth_occurrence(n=-1)[0].date()), display=True)
        elif selection == 3:
            params["min_listen_time"] = force_int("Please enter a minimum listening time: (milliseconds)", default=0,
                                                  min_int=0, display=True)
        else:
            selection = -1
        print()


# UI MAIN PATTERN
print("Welcome to Brit's SpotifyDataAnalysisPy")
print("Loading Users...")

spotify = SpotifyAnalysis("C:/Users/britt/Desktop/Java Projects/SpotifyDataAnalysisPy/SpotifyAnalysis/SpotifyData")
params = {"current_user": "Brittan", "start": None, "end": None, "min_listen_time": 0}

print()
menu_selection = 0
while menu_selection >= 0:
    print_title("Main Menu - Please select an option:", 90)
    menu_selection = list_options(["View Statistics and Summaries", "Create Charts", "Create Plots",
                                   "Settings", "Exit"])
    if menu_selection == 0:
        statistics_and_summaries()
    elif menu_selection == 1:
        create_charts()
    elif menu_selection == 2:
        create_plots()
    elif menu_selection == 3:
        settings()
    else:
        menu_selection = -1
    print()

# DEMO 1
# top = top_n_entries(n=5, master_entries=["black country, new road"], entry_type="album")
# plot_over_time(entries=top, entry_type="album")



for string in top_n(n=1000, entry_type="artist")["artist"].tolist():
    if len(string.split()) >= 3:
        print(string)