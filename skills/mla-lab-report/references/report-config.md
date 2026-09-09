# Optional `mla-report-config.json`

Create this file once in the repository when the group's stable submission details are known. Keep it out of public repositories if it contains personal data.

```json
{
  "group_prefix": "Group01_CLC01",
  "submitting_member": {
    "name": "Your full name",
    "student_id": "Your student ID"
  },
  "course": "Machine Learning Applications (MLA)",
  "academic_year": "2026-2027"
}
```

The `group_prefix` is the exact portion before `_Lab<N>` in the required filename. The skill may read this file to avoid asking for the same metadata every week, but it must still verify the lab number and any extra fields required by that week's official instruction PDF.

Do not put passwords, API keys, or other secrets in this file. Do not commit real student identity data unless the repository is intended to contain it.
