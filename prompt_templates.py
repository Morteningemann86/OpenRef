PROMPT_TEMPLATES = {
    "Municipality General Meeting": {
        "system": "Write a JSON structure for a formal municipality meeting summary. The response must be valid JSON that follows this structure: {\"Meeting Overview\":\"Description of the general agenda and purpose\", \"Key Decisions\":\"List of major decisions made\", \"Action Items\":\"List of specific tasks and assignments\", \"Next Steps\":\"Follow-up actions and future planning\"}",
        "shot_example": '''
# 🏛️ Municipality General Meeting Summary

## 📌 Agenda  
- **Introduction**: Overview of key topics  
- **Department Updates**: Progress reports from various teams  
- **Ongoing Projects**: Status of municipal initiatives  
- **Community Concerns**: Issues raised and potential solutions  
- **Action Points**: Key takeaways and next steps  

## 🏗️ Key Decisions  
- **Decision 1**: [Description]  
- **Decision 2**: [Description]  

## 📅 Next Steps  
- **[Task]** assigned to **[Person/Department]** (Deadline: **[Date]**)  
'''
    },
    "Municipality Budget Meeting": {
        "system": "Write a JSON structure for a municipal budget meeting summary. The response must be valid JSON that follows this structure: {\"Budget Overview\":\"Summary of total budget and allocations\", \"Financial Discussions\":\"Key points from budget deliberations\", \"Approved Changes\":\"List of approved budget modifications\", \"Financial Action Items\":\"Specific financial tasks and responsibilities\"}",
        "shot_example": '''
# 💰 Municipality Budget Meeting Summary

## 📊 Budget Overview  
- **Total Budget**: [Amount]  
- **Key Allocations**:  
  - Department A: [Amount]  
  - Department B: [Amount]  
  - Infrastructure: [Amount]  

## 🔍 Discussion Points  
- **Revenue Sources**: Taxation, grants, funding opportunities  
- **Cost Reductions**: Areas for efficiency improvements  
- **Future Investments**: Key projects requiring funding  

## ✅ Approved Budget Changes  
- **[Change]**: [Details]  

## 🏁 Action Items  
- **[Task]** assigned to **[Person/Department]** (Deadline: **[Date]**)  
'''
    },
    "Municipality Citizen Engagement Meeting": {
        "system": "Write a JSON structure for a citizen engagement meeting summary. The response must be valid JSON that follows this structure: {\"Meeting Participants\":\"Overview of attendees and roles\", \"Community Topics\":\"Key issues raised by citizens\", \"Proposed Solutions\":\"Suggested resolutions and approaches\", \"Community Action Items\":\"Specific follow-up tasks and commitments\"}",
        "shot_example": '''
# 🏘️ Citizen Engagement Meeting Summary

## 👥 Participants  
- **Municipal Representatives**: [Names/Roles]  
- **Community Members**: [Number of attendees]  

## 📢 Key Topics Discussed  
- **Issue 1**: [Description]  
- **Issue 2**: [Description]  

## 🔄 Proposed Solutions  
- **Solution A**: [Details]  
- **Solution B**: [Details]  

## 📝 Action Points  
- **Follow-up on [Issue]** (Responsible: **[Person/Department]**, Deadline: **[Date]**)  
- **Next Meeting Scheduled**: **[Date]**  
'''
    }
}
