from pydantic import BaseModel, ConfigDict, Field


class LeadershipMember(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    role: str
    linkedin_url: str | None


class CompanyIntelligence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_overview: str
    target_audience: str
    contact_points: list[str]
    leadership_team: list[LeadershipMember]
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
    )


class CompanyLead(CompanyIntelligence):
    model_config = ConfigDict(extra="forbid")

    source_urls: list[str]
    pages_analyzed: list[str]
    crawl_status: str
    extraction_warnings: list[str]