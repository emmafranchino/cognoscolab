import pandas as pd
import datetime
import numpy as np
from collections import defaultdict
from config import QUALTRICS_WORDS_FIELDS, WORDS_ORDER_VALENCE_TASK


def get_trimmed_df(raw_data, date, participant_label):
    raw_data = raw_data[2:]
    # remove all the columns that are not needed and the first two rows which are 'index' of the columns
    raw_data = raw_data.drop(
        columns=[
            "EndDate",
            "Status",
            "IPAddress",
            "RecordedDate",
            "ResponseId",
            "RecipientLastName",
            "RecipientFirstName",
            "RecipientEmail",
            "ExternalReference",
            "LocationLatitude",
            "LocationLongitude",
            "DistributionChannel",
            "UserLanguage",
        ]
    )[2:]

    # convert the progress values into integers
    progress = raw_data["Progress"].values.astype(int)

    # keep only the participations with more than 65% progress
    raw_data = raw_data[(progress > 65) & (raw_data["QID3"] == "1")].reset_index(
        drop=True
    )

    if date == None:
        data_population = raw_data.reset_index(drop=True).copy()

    else:
        data_population = (
            raw_data[pd.to_datetime(raw_data["StartDate"]).dt.date == date]
            .reset_index(drop=True)
            .copy()
        )

    data_population["participant_id"] = [
        f"{participant_label}_{i:03d}" for i in range(1, len(data_population) + 1)
    ]

    return data_population


def get_associations(data_population):
    # make lists for the questions ID of the associations task questions and their text labels
    keys_associations = []

    keys_labels = []

    # extract the questions ID and labels from the fields defined above
    for j in range(1, 8):
        for i, field in enumerate(QUALTRICS_WORDS_FIELDS):
            for a in range(1, 4):
                keys_associations.append(f"{j}_QID71_{i+1}_{a}")
                keys_labels.append(field[j - 1])

    # create a dictionary to store the associations for each keyword
    keywords_dict = defaultdict(list)

    for key, value in data_population.items():
        # 126 = 42 keywords * 3 associations
        for i in range(126):
            if key == keys_associations[i]:
                if keys_labels[i] not in keywords_dict:
                    keywords_dict[keys_labels[i]] = []
                keywords_dict[keys_labels[i]].append(value.tolist())

    series_associations_df = pd.DataFrame.from_dict(keywords_dict)

    # reorder the rows of df_keywords according to valence_keywords_order
    series_associations_df = series_associations_df[WORDS_ORDER_VALENCE_TASK]

    # transpose the DataFrame to have cue words as rows and associations as columns
    df_transposed = series_associations_df.T.reset_index()

    df_transposed.columns = [
        "cue_word",
        "association_1",
        "association_2",
        "association_3",
    ]

    # set as index the cue words
    associations_task_df = df_transposed.set_index("cue_word")

    # explode the associations into separate rows and reset the index
    associations_task_df = associations_task_df.apply(pd.Series.explode).reset_index()

    # add the participant_id column multiplying the participant IDs for each cue word
    associations_task_df["participant_id"] = np.tile(
        data_population["participant_id"].values, len(WORDS_ORDER_VALENCE_TASK)
    )

    associations_task_df = associations_task_df[
        [
            "participant_id",
            "cue_word",
            "association_1",
            "association_2",
            "association_3",
        ]
    ]

    return associations_task_df


def get_valences(data_population):
    # IDs of the questions of the valence task
    q_ids = ["Q20", "QID63", "QID64", "Q15", "Q16"]

    # creation of list to store the keys and values of the valence task
    keys_valence = []
    values_valence = []

    # extraction of all the questions IDs (keys) and the values based on the q_id and their indices (see expalanation above)
    for key, value in data_population.items():
        for j in range(1, 10):
            for q_id in q_ids:
                for i in range(5):
                    if key == f"{j}_{q_id}#{i}_1":
                        # check if the column is not empty (if there are all NaN values)
                        if not data_population[key].iloc[:].isna().all():
                            keys_valence.append(key)
                            values_valence.append(data_population[key])

    # create a dictionary with the keys (q_ids) and values extracted from the valence task
    valence_dict = dict(zip(keys_valence, values_valence))

    # create lists for questions IDs labels
    labels = []

    # take each of the cue word from valence_keywords_order and create labels for each association in the valence task
    for word_key in WORDS_ORDER_VALENCE_TASK:
        for i in range(4):
            labels.append(f"{word_key}_association_{i}")

    # create a dict with the list of all questions IDs (previously created) and the label associated to each of them
    id_to_label = dict(zip(keys_valence, labels))

    # assign to the question IDs the question label so that i understand to what the scores are associeted
    labeled_scores = pd.DataFrame.from_dict(valence_dict, orient="index").T
    labeled_scores.rename(columns=id_to_label, inplace=True)

    # melt to long format: each row is one valence_key per participant
    melted_valence_df = labeled_scores.melt(var_name="key", value_name="valence")

    # assign an index to the df, based on the number of association (0 to 4) so that i can make separate columns for each of them
    melted_valence_df["index"] = (
        melted_valence_df["key"].str.extract(r"association_(\d)").astype(int)
    )
    melted_valence_df["valence"] = pd.to_numeric(
        melted_valence_df["valence"], errors="coerce"
    ).astype("Int64")

    # adds a new column ("row") to valence_df that tells the position of each row within its group based on the "index" column
    melted_valence_df["row"] = melted_valence_df.groupby("index").cumcount()

    # reshapes to wide format using as index the row as columns headers the index and as values the key and valence
    valence_task_df = melted_valence_df.pivot(
        index="row", columns="index", values=["key", "valence"]
    )

    # create names for the columns based on their number of association and values
    valence_task_df.columns = [
        f"{col}_association_{i}" for col, i in valence_task_df.columns
    ]

    # reset the index without reassing it to a new variable
    valence_task_df.reset_index(drop=True, inplace=True)

    return valence_task_df


def associations_valences_df(associations_task_df, valence_task_df):
    # concatenate the df with the labels for the valence task and the df of the association taks (previously created)
    associations_valence_df = pd.concat([associations_task_df, valence_task_df], axis=1)

    columns_order = [
        "participant_id",
        "key_association_0",
        "cue_word",
        "valence_association_0",
        "key_association_1",
        "association_1",
        "valence_association_1",
        "key_association_2",
        "association_2",
        "valence_association_2",
        "key_association_3",
        "association_3",
        "valence_association_3",
    ]

    # redorder the df based on columns_order and remove the columns contaning the labels of the questions IDs in the valence task
    associations_valence_df = associations_valence_df[columns_order].drop(
        columns=[
            "key_association_0",
            "key_association_1",
            "key_association_2",
            "key_association_3",
        ]
    )

    # rename the column related to the cue word in the valence task
    associations_valence_df = associations_valence_df.rename(
        columns={"valence_association_0": "valence_cue_word"}
    )

    return associations_valence_df


def create_associations_valence_df(
    raw_data_test, date=datetime.date(2026, 3, 2), participant_label="teacher"
):
    data_population = get_trimmed_df(raw_data_test, date, participant_label)
    associations_df = get_associations(data_population)
    df_valences = get_valences(data_population)
    associations_valences_teachers = associations_valences_df(
        associations_df, df_valences
    )
    return associations_valences_teachers
