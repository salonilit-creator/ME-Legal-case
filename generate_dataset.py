import random
import faker

# Generate a random case record
class Case:
    def __init__(self, case_type, case_id, party_a, party_b, date_filed):
        self.case_type = case_type
        self.case_id = case_id
        self.party_a = party_a
        self.party_b = party_b
        self.date_filed = date_filed

    def __str__(self):
        return f"{self.case_type} Case (ID: {self.case_id}): {self.party_a} vs {self.party_b} filed on {self.date_filed}"

# List of legal case types
case_types = ["Criminal", "Civil", "Commercial", "Family", "Intellectual Property", "Tax", "Environmental", "Labor", "Bankruptcy"]

# Initialize Faker
fake = faker.Faker()

# Generate a dataset of case records
def generate_dataset(num_cases):
    dataset = []
    for _ in range(num_cases):
        case_type = random.choice(case_types)
        case_id = fake.uuid4()
        party_a = fake.name()
        party_b = fake.name()
        date_filed = fake.date_this_decade()
        case_record = Case(case_type, case_id, party_a, party_b, date_filed)
        dataset.append(case_record)
    return dataset

# Generate 100 case records as an example
if __name__ == '__main__':
    case_records = generate_dataset(100)
    for case in case_records:
        print(case)