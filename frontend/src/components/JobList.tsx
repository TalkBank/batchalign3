import { useFilteredJobs } from "../state";
import { JobCard } from "./JobCard";
import { EmptyState } from "./EmptyState";

export function JobList() {
  const items = useFilteredJobs();
  if (items.length === 0) return <EmptyState />;
  return (
    <div className="flex flex-col gap-3">
      {items.map((job) => (
        <JobCard key={job.job_id} job={job} />
      ))}
    </div>
  );
}
