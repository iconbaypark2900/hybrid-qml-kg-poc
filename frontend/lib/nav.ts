export type NavSection = {
  heading: string;
  items: { href: string; label: string }[];
};

export const navSections: NavSection[] = [
  {
    heading: "Journey",
    items: [
      { href: "/", label: "Start" },
      { href: "/experiments", label: "Experiment" },
      { href: "/validation", label: "Validation" },
      { href: "/visualization", label: "Visual" },
      { href: "/settings", label: "Settings" },
    ],
  },
];
