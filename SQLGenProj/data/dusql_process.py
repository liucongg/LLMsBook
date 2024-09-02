# -*- coding: utf-8 -*-
import argparse
import json
import os
import requests
from tqdm import tqdm
import copy
from translation_service import Translation


class DusqlDataSet:
    def __init__(self, home_path, translation_model_path):
        self.home_path = home_path
        self.translation = Translation(translation_model_path)

    def translation_service(self, text):
        result = self.translation.translate(text)
        en_query = result.strip(".").replace(".", " ").replace(",", " ")
        en_query = en_query.replace(" ", "_")
        return en_query

    @staticmethod
    def load_data(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def get_column_types(self, col_type):
        type_dict = {"number": "INTEGER", "text": "VARCHAR(50)", "binary": "BINARY", "time": "DATETIME",
                     "data": "DATETIME"}

        if col_type in type_dict:
            return type_dict[col_type]
        return "VARCHAR(50)"

    def get_sqlite(self):
        result = {}
        with open(os.path.join(self.home_path, "new_schema.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                whole_sql_info = []
                sample = json.loads(line)
                db_id = sample["db_id"]
                columns_en = sample['column_en']
                table_en = sample['table_en']
                joined_info = sample['joined_info']
                for table_name, columns in sample["table_info"].items():
                    is_first = True
                    # table_name_en = table_en[table_name]
                    table_info = f"CREATE TABLE {table_name} "
                    column_info = []
                    for column in columns:
                        column_name_zh, column_type = column
                        column_name_en = columns_en[column_name_zh]
                        column_name_en = " ".join(column_name_en.split("_"))
                        column_sql_type = self.get_column_types(column_type)
                        "  product_id INTEGER PRIMARY KEY, -- Unique ID for each product"
                        if is_first:
                            column_info.append(f"  {column_name_zh} {column_sql_type} PRIMARY KEY, -- {column_name_en}")
                            is_first = False
                        else:
                            column_info.append(f"  {column_name_zh} {column_sql_type}, -- {column_name_en}")
                    one_table_info = table_info + "(\n" + "\n".join(column_info) + "\n);"
                    whole_sql_info.append(one_table_info)
                joined_part = []
                for one_join in joined_info:
                    a, b = one_join
                    table_name_zh_a, column_name_zh_a = a[0], a[1]
                    table_name_zh_b, column_name_zh_b = b[0], b[1]
                    one_join_info = f"-- {table_name_zh_a}.{column_name_zh_a} can be joined with {table_name_zh_b}.{table_name_zh_b}"
                    joined_part.append(one_join_info)
                whole_sql_info.append("\n".join(joined_part))
                result[db_id] = {"sqlite": "\n".join(whole_sql_info), "columns_en": columns_en, "table_en": table_en}
        return result

    def trans_schema(self):
        db_schema = self.load_data(os.path.join(self.home_path, "db_schema.json"))
        new_schema = []
        for one_db in tqdm(db_schema):
            db_id = one_db['db_id']
            table_info = {}
            column_en = {}
            table_en = {}
            for i, column_info in enumerate(tqdm(one_db['column_names'][1:])):
                table_id, column_name = column_info
                column_name_en = self.translation_service(column_name)
                column_en[column_name] = column_name_en
                column_type = one_db['column_types'][i]
                table_name = one_db['table_names'][table_id]
                table_name_en = self.translation_service(table_name)
                table_en[table_name] = table_name_en
                if table_name in table_info:
                    table_info[table_name].append([column_name, column_type])
                else:
                    table_info[table_name] = [[column_name, column_type]]
            foreign_keys = one_db["foreign_keys"]
            joined_info = []
            for keys in foreign_keys:
                a, b = one_db['column_names'][keys[0]], one_db['column_names'][keys[1]]
                table_name_a, column_name_a = one_db['table_names'][a[0]], a[1]
                table_name_b, column_name_b = one_db['table_names'][b[0]], b[1]
                joined_info.append(([table_name_a, column_name_a], [table_name_b, column_name_b]))
            schema_info = {"db_id": db_id, "table_info": table_info, "joined_info": joined_info,
                           "column_en": column_en, "table_en": table_en}
            new_schema.append(schema_info)
        return new_schema

    def make_llm_data(self, file_name, save_name, sqlite_info_name="sqlite_info_zh.json"):
        llm_data = []
        with open(os.path.join(self.home_path, sqlite_info_name), 'r', encoding="utf-8") as f:
            sqlite_info = json.load(f)
        with open(os.path.join(self.home_path, file_name), 'r', encoding="utf-8") as f:
            samples = json.load(f)
            for sample in tqdm(samples):
                db_id = sample['db_id']
                question = sample['question']
                sql_query_zh = sample["sql_query"]
                sqlite_query = sqlite_info[db_id]["sqlite"]
                prompt = f"""### Instructions:
Your task is convert a question into a SQL query, given a Postgres database schema.
Adhere to these rules:
- **Deliberately go through the question and database schema word by word** to appropriately answer the question
- **Use Table Aliases** to prevent ambiguity. For example, `SELECT table1.col1, table2.col1 FROM table1 JOIN table2 ON table1.id = table2.id`.
- When creating a ratio, always cast the numerator as float

### Input:
Generate a SQL query that answers the question `{question}`.
This query will run on a database whose schema is represented in this string:
{sqlite_query}

### Response:
Based on your instructions, here is the SQL query I have generated to answer the question `{question}`:
```sql
"""

                # prompt = instruction.format(table_info=sqlite_info, input=question)
                output = sql_query_zh
                llm_data.append({"input": prompt,
                                 "output": output})

        with open(os.path.join(self.home_path, save_name), 'w', encoding="utf-8") as fout:
            fout.writelines("\n".join([json.dumps(one, ensure_ascii=False) for one in llm_data]))


def parse_args():
    parser = argparse.ArgumentParser(description='llama2-7B QLoRA')
    parser.add_argument('--sql_home_path', type=str, default="./dusql/", help='dusql数据保存地址')
    parser.add_argument('--translation_model_path', type=str, default="", help='翻译模型地址')
    return parser.parse_args()


if __name__ == '__main__':
    home_path = "./dusql/"
    translation_model_path = ""
    data = DusqlDataSet(home_path, translation_model_path)
    result = data.get_sqlite()
    with open(os.path.join(home_path, "sqlite_info_zh.json"), 'w', encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    data.make_llm_data("dev.json", "llm_dev_zh.json")
    data.make_llm_data("train.json", "llm_train_zh.json")
