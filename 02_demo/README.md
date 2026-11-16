# Demo

One demo, two sides. Serve the repository folder and open it in a browser:

```sh
python3 -m http.server 8817     # from the repository root
```

Then go to `http://127.0.0.1:8817/`.

| Page | What it is |
|---|---|
| `index.html` | The start page: choose a side |
| `canada/index.html` | **Canada side.** A four-step planner for 10 Saskatchewan districts and 8 crops. Its numbers are in `canada/data.json`. |
| `us/index.html` | **US side.** One self-contained page: county map, farm priorities, trial evidence, model check and sources. It also opens by double-clicking the file, with no internet needed. |

Both sides carry a Canada / US switch at the top. Entries stay in your browser.

`canada/data.json` is written by `04_reproduce/saskatchewan/scripts/export_demo.py`; `us/index.html` is built by `04_reproduce/demo/build_demo.py`.
