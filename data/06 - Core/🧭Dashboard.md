## 📅 Today’s Note
```dataview
list
from "01 - Daily"
where file.name = dateformat(date(today), "yyyy-MM-dd")
```
## 🧠 Recent Notes  
```dataview
list
from "02 - Notes"
sort file.mtime desc
limit 10
```
## 📚 Recent References
```dataview
list
from "04 - Reference"
sort file.mtime desc
limit 10
```
## 🚧 Ongoing Projects
```dataview
list
from "03 - Projects"
where contains(Tags, "#in-progress")
sort file.mtime desc
limit 10
```
## 🔁 Random Past Notes

```dataview
list
from "02 - Notes"
where file.mtime < date(today) - dur(30 days)
sort file.mtime desc
limit 3
```

