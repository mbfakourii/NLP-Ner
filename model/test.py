def test_model(nlp2, df):
    # Drop Doc_ID
    df.drop(["Doc_ID"], axis=1, inplace=True)

    # Crate Column Tag
    df['tag'] = 'O'

    # Crate Line And Location Words
    line = ""
    locations = []

    # Check Test Lines
    for i, row in df.iterrows():
        if row.Word == ".":
            # test
            doc2 = nlp2(str(line))

            # Search Words In Test Data
            for ent in doc2.ents:
                for element in locations:
                    if ent.text in element:
                        # Save Label In Tag
                        df.at[element.get(ent.text), "tag"] = ent.label_
                        break

            # Clear Line And Location
            line = ""
            locations = []
        else:
            # Add Location And Set Line From Words
            locations.append({row.Word: i})
            if line == "":
                line = str(row.Word)
            else:
                line = line + " " + str(row.Word)

    # Drop Word Column And Save Data
    df.drop(["Word"], axis=1, inplace=True)
    df.to_csv("output/submission.csv", index=False)