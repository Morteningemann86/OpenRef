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
        "system": "Write a JSON structure for a citizen engagement meeting summary with speaker attribution. The response must be valid JSON that follows this structure: {\"Meeting Participants\":\"Overview of number of speakers identified\", \"Community Topics\":\"Key issues raised by speakers\", \"Proposed Solutions\":\"Suggested resolutions with speaker attribution\", \"Community Action Items\":\"Specific follow-up tasks and commitments\"}",
        "shot_example": '''
# 🏘️ Citizen Engagement Meeting Summary

## 👥 Participants  
- Number of speakers identified in discussion

## 📢 Key Topics Discussed  
- Key points raised by each speaker
- Important discussion threads
- Main concerns and suggestions

## 🔄 Proposed Solutions  
- Solutions presented during discussion
- Collaborative suggestions from multiple speakers
- Areas of consensus

## 📝 Action Points  
- Follow-up tasks identified
- Next steps agreed upon
- Timeline for implementation
'''
    }
}
