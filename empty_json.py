import json

# Step 1: Read the JSON data from the file
with open("games.json", "r") as file:
    data = json.load(file)


data["strategy"] = []

# Step 3: Write the modified data back to the JSON file
with open("games.json", "w") as file:
    json.dump(data, file, indent=4)  # Optionally, you can format the JSON for better readability

# All items within the "strategy" key have been deleted from the JSON data in the file