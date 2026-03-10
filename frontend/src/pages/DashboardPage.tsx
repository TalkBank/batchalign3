import { StatsRow } from "../components/StatsRow";
import { JobList } from "../components/JobList";
import { ServerFilterTabs } from "../components/ServerFilterTabs";
import { useStore } from "../state";

export function DashboardPage() {
  const serverFilter = useStore((s) => s.serverFilter);
  const setServerFilter = useStore((s) => s.setServerFilter);

  return (
    <div>
      <ServerFilterTabs selected={serverFilter} onSelect={setServerFilter} />
      <StatsRow />
      <div className="mt-6">
        <JobList />
      </div>
    </div>
  );
}
