import json

def fix_notebook(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    changes = []

    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue

        src = cell['source']
        is_list = isinstance(src, list)
        full = ''.join(src) if is_list else src
        original = full

        # Fix: strip result_df (DataFrame) before storing in LangGraph state
        old = "    return {'messages': [AIMessage(content=text)], 'sql_result': result}"
        new = (
            "    # Strip DataFrame — not msgpack-serializable by LangGraph checkpoint\n"
            "    result_safe = {k: v for k, v in result.items() if k != 'result_df'}\n"
            "    return {'messages': [AIMessage(content=text)], 'sql_result': result_safe}"
        )

        if old in full and new not in full:
            full = full.replace(old, new)
            changes.append(cell.get('id', '?'))

        if full != original:
            cell['source'] = full.splitlines(keepends=True) if is_list else full

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    return changes


paths = [
    r'E:\CODING\edu_test\dmu_final_tavily.ipynb',
    r'E:\CODING\ed\dmu_final_tavily.ipynb',
]

for path in paths:
    changes = fix_notebook(path)
    status = f'patched cells: {changes}' if changes else 'already fixed'
    print(f'{path}\n  -> {status}\n')
