import pymongo
from numpy.random import choice
from utils.printer import dump_data

"""
This file is the interfere with database
"""


class Database:
    """Database setting"""

    signature_collection = "signature"
    similarity_collection = "similarity"
    argdef_collection = "api_args"

    def __init__(self) -> None:
        pass

    def database_config(self, host, port, database_name):
        self.DB = pymongo.MongoClient(host=host, port=port)[database_name]

    def index_name(self, api_name, arg_name):
        record = self.DB[self.signature_collection].find_one({"api": api_name})
        if record is None:
            print(api_name + " no signature")
            return None
        arg_names = record["args"]
        for idx, name in enumerate(arg_names):
            if name == arg_name:
                return f"parameter:{idx}"
        return None

    def select_rand_over_db(self, api_name, arg_name):
        if api_name not in self.DB.list_collection_names():
            return None, False
        record = self.DB[self.signature_collection].find_one({"api": api_name})
        if record is None:
            return None, False
        arg_names = record["args"]
        if arg_name.startswith("parameter:"):
            index = int(arg_name[10:])
            if index >= len(arg_names):
                return None, False
            arg_name = arg_names[index]

        sim_dict = self.DB[self.similarity_collection].find_one(
            {"api": api_name, "arg": arg_name}
        )
        if sim_dict is None:
            return None, False
        APIs = sim_dict["APIs"]
        probs = sim_dict["probs"]
        if len(APIs) == 0:
            return None, False
        target_api = choice(APIs, p=probs)
        # compare the time of 2 operations
        idx_name = self.index_name(target_api, arg_name)
        if idx_name is None:
            return None, False
        select_data = self.DB[target_api].aggregate(
            [
                {
                    "$match": {
                        "$or": [
                            {
                                arg_name: {"$exists": True},
                            },
                            {idx_name: {"$exists": True}},
                        ]
                    }
                },
                {"$sample": {"size": 1}},
            ]
        )
        if not select_data.alive:
            # not found any value in the (target_api, arg_name)
            print(f"ERROR IN SIMILARITY: {target_api}, {api_name}")
            return None, False
        select_data = select_data.next()
        if arg_name in select_data.keys():
            return select_data[arg_name], True
        else:
            return select_data[idx_name], True

    def get_rand_record(self, api_name):
        record = self.DB[api_name].aggregate([{"$sample": {"size": 1}}])
        if not record.alive:
            print(f"NO SUCH API: {api_name}")
            assert 0
        record = record.next()
        record.pop("_id")
        assert "_id" not in record.keys()
        return record

    def get_all_records(self, api_name):
        if api_name not in self.DB.list_collection_names():
            print(f"NO SUCH API: {api_name}")
            return []
        temp = self.DB[api_name].find({}, {"_id": 0})
        records = []
        for t in temp:
            assert "_id" not in t.keys()
            records.append(t)
        return records

    def get_argdef(self, api_name):
        record = self.DB[self.argdef_collection].find_one(
            {"api": api_name}, {"_id": 0}
        )
        if record is None:
            print(f"NO API_ARGS FOR: {api_name}")
            dump_data(f"NO API_ARGS FOR: {api_name}\n", "db-error.txt", "a")
            assert 0
        return record["args"]

    def get_signature(self, api_name):
        record = self.DB[self.signature_collection].find_one(
            {"api": api_name}, {"_id": 0}
        )
        if record is None:
            print(f"NO SIGNATURE FOR: {api_name}")
            assert 0
        return record["args"]

    def add_record(self, api_name, record):
        """
        Add one record of <api_name> into the Database
        """
        print(api_name, record)
        try:
            self.DB[api_name].insert_one(record.copy())
        except Exception:
            dump_data(f"{api_name} {record}\n", "database-log.txt", "a")

    def add_records(self, api_name, record):
        """
        Add many records of <api_name> into the Database
        """
        self.DB[api_name].insert_many(record)
        print(api_name, record)

    def add_signature(self, api_name, signature):
        """
        Add signature of <api_name> into the database
        """
        data = {"api": api_name, "args": signature.copy()}
        self.DB[self.signature_collection].insert_one(data)

    def add_argdef(self, api_name, argdef):
        """
        Add argdef of <api_name> into the database
        """
        data = {"api": api_name, "args": argdef.copy()}
        self.DB[self.argdef_collection].insert_one(data)

    def delete_all_argdef(self, api_name):
        self.DB[self.argdef_collection].delete_many({"api": api_name})

    @staticmethod
    def get_api_list(DB, start_str, sorted=True):
        api_list = []
        for name in DB.list_collection_names():
            if name.startswith(start_str):
                api_list.append(name)
        if sorted:
            api_list.sort()
        return api_list


class TorchDB(Database):
    def __init__(self) -> None:
        super().__init__()

    def get_api_list(self):
        self.api_list = super().get_api_list(self.DB, "torch.")
        return self.api_list


class TFDB(Database):
    def __init__(self) -> None:
        super().__init__()

    def get_api_list(self):
        self.api_list = super().get_api_list(self.DB, "tf.")
        return self.api_list


class JaxDB(Database):
    def __init__(self) -> None:
        super().__init__()

    def get_api_list(self):
        self.api_list = super().get_api_list(self.DB, "jax.")
        return self.api_list

class OneflowDB(Database):
    def __init__(self) -> None:
        super().__init__()

    def get_api_list(self):
        self.api_list = super().get_api_list(self.DB, "oneflow.")
        return self.api_list

"""
Database for each library
NOTE:
You must config the database by using `database_config(host, port, name)` before use!!!
Like TFDatabase.database_config("127.0.0.1", 27109, "tftest")
"""
TorchDatabase = TorchDB()
TFDatabase = TFDB()
JaxDatabase = JaxDB()
OneflowDatabase = OneflowDB()
